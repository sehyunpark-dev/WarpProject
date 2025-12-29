import warp as wp
from typing import TYPE_CHECKING

from solvers.base_solver import Solver
from core.dim3d.mac_grid_3d import MACGrid3D, lookup_float, sample_float, sample_scalar, sample_velocity, sample_u, sample_v, sample_w, compute_divergence, compute_neighbor_pressure

if TYPE_CHECKING:
    from core.dim3d.simulation_3d import EmitterData3D, MaskData3D, InitialVelocityData3D


#######################################################################
# External Force Kernels
#######################################################################

@wp.kernel
def apply_emitters_kernel(
    smoke: wp.array3d(dtype=float),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    emitter_centers: wp.array1d(dtype=wp.vec3),
    emitter_radii: wp.array1d(dtype=float),
    emitter_heights: wp.array1d(dtype=float),
    emitter_velocities: wp.array1d(dtype=wp.vec3),
    emitter_smoke_amounts: wp.array1d(dtype=float),
    num_emitters: int,
    dx: float,
    dt: float
):
    """
    Apply multiple emitters: inject smoke and velocity.
    Each emitter is a cylindrical region (radius in XZ plane, height in Y).
    """
    i, j, k = wp.tid()

    nx = smoke.shape[0]
    ny = smoke.shape[1]
    nz = smoke.shape[2]

    if i >= nx or j >= ny or k >= nz:
        return

    # Cell center in world coordinates
    px = (float(i) + 0.5) * dx
    py = (float(j) + 0.5) * dx
    pz = (float(k) + 0.5) * dx

    # Check all emitters
    for e in range(num_emitters):
        center = emitter_centers[e]
        radius = emitter_radii[e]
        height = emitter_heights[e]
        vel = emitter_velocities[e]
        amount = emitter_smoke_amounts[e]

        # Distance in XZ plane (cylindrical check)
        dist_xz = wp.sqrt((px - center[0]) * (px - center[0]) + (pz - center[2]) * (pz - center[2]))

        # Check if within cylinder: radius in XZ, height in Y
        if dist_xz < radius and wp.abs(py - center[1]) < height * 0.5:
            # Inject smoke
            smoke[i, j, k] = amount

            # Apply velocity to adjacent faces
            # u-faces (x velocity)
            if wp.abs(vel[0]) > 0.0:
                u[i, j, k] = vel[0]
                if i + 1 < nx + 1:
                    u[i + 1, j, k] = vel[0]

            # v-faces (y velocity)
            if wp.abs(vel[1]) > 0.0:
                v[i, j, k] = vel[1]
                if j + 1 < ny + 1:
                    v[i, j + 1, k] = vel[1]

            # w-faces (z velocity)
            if wp.abs(vel[2]) > 0.0:
                w[i, j, k] = vel[2]
                if k + 1 < nz + 1:
                    w[i, j, k + 1] = vel[2]


@wp.kernel
def apply_initial_velocity_kernel(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    iv_centers: wp.array1d(dtype=wp.vec3),
    iv_half_sizes: wp.array1d(dtype=wp.vec3),
    iv_velocities: wp.array1d(dtype=wp.vec3),
    num_regions: int,
    nx: int,
    ny: int,
    nz: int,
    dx: float
):
    """
    Apply initial velocity to specified box regions.
    Called once at simulation start (in reset()).
    """
    i, j, k = wp.tid()

    # Process u-faces (shape: nx+1, ny, nz)
    if i <= nx and j < ny and k < nz:
        px_u = float(i) * dx
        py_u = (float(j) + 0.5) * dx
        pz_u = (float(k) + 0.5) * dx

        for r in range(num_regions):
            center = iv_centers[r]
            half_size = iv_half_sizes[r]
            vel = iv_velocities[r]

            # Check if inside box
            if (wp.abs(px_u - center[0]) < half_size[0] and
                wp.abs(py_u - center[1]) < half_size[1] and
                wp.abs(pz_u - center[2]) < half_size[2]):
                u[i, j, k] = vel[0]

    # Process v-faces (shape: nx, ny+1, nz)
    if i < nx and j <= ny and k < nz:
        px_v = (float(i) + 0.5) * dx
        py_v = float(j) * dx
        pz_v = (float(k) + 0.5) * dx

        for r in range(num_regions):
            center = iv_centers[r]
            half_size = iv_half_sizes[r]
            vel = iv_velocities[r]

            # Check if inside box
            if (wp.abs(px_v - center[0]) < half_size[0] and
                wp.abs(py_v - center[1]) < half_size[1] and
                wp.abs(pz_v - center[2]) < half_size[2]):
                v[i, j, k] = vel[1]

    # Process w-faces (shape: nx, ny, nz+1)
    if i < nx and j < ny and k <= nz:
        px_w = (float(i) + 0.5) * dx
        py_w = (float(j) + 0.5) * dx
        pz_w = float(k) * dx

        for r in range(num_regions):
            center = iv_centers[r]
            half_size = iv_half_sizes[r]
            vel = iv_velocities[r]

            # Check if inside box
            if (wp.abs(px_w - center[0]) < half_size[0] and
                wp.abs(py_w - center[1]) < half_size[1] and
                wp.abs(pz_w - center[2]) < half_size[2]):
                w[i, j, k] = vel[2]


@wp.kernel
def apply_masks_kernel(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    mask_centers: wp.array1d(dtype=wp.vec3),
    mask_radii: wp.array1d(dtype=float),
    num_masks: int,
    nx: int,
    ny: int,
    nz: int,
    dx: float
):
    """
    Apply obstacle masks: set velocity to zero inside solid regions (spherical).
    """
    i, j, k = wp.tid()

    # Process u-faces (shape: nx+1, ny, nz)
    if i <= nx and j < ny and k < nz:
        px_u = float(i) * dx
        py_u = (float(j) + 0.5) * dx
        pz_u = (float(k) + 0.5) * dx

        for m in range(num_masks):
            center = mask_centers[m]
            radius = mask_radii[m]
            dist = wp.sqrt((px_u - center[0]) ** 2.0 + (py_u - center[1]) ** 2.0 + (pz_u - center[2]) ** 2.0)
            if dist < radius:
                u[i, j, k] = 0.0

    # Process v-faces (shape: nx, ny+1, nz)
    if i < nx and j <= ny and k < nz:
        px_v = (float(i) + 0.5) * dx
        py_v = float(j) * dx
        pz_v = (float(k) + 0.5) * dx

        for m in range(num_masks):
            center = mask_centers[m]
            radius = mask_radii[m]
            dist = wp.sqrt((px_v - center[0]) ** 2.0 + (py_v - center[1]) ** 2.0 + (pz_v - center[2]) ** 2.0)
            if dist < radius:
                v[i, j, k] = 0.0

    # Process w-faces (shape: nx, ny, nz+1)
    if i < nx and j < ny and k <= nz:
        px_w = (float(i) + 0.5) * dx
        py_w = (float(j) + 0.5) * dx
        pz_w = float(k) * dx

        for m in range(num_masks):
            center = mask_centers[m]
            radius = mask_radii[m]
            dist = wp.sqrt((px_w - center[0]) ** 2.0 + (py_w - center[1]) ** 2.0 + (pz_w - center[2]) ** 2.0)
            if dist < radius:
                w[i, j, k] = 0.0


#######################################################################
# Advection Kernels
#######################################################################

@wp.kernel
def advect_u(grid: MACGrid3D, dt: float):
    i, j, k = wp.tid()

    # u field size: (nx+1, ny, nz)
    if i >= grid.nx + 1 or j >= grid.ny or k >= grid.nz:
        return

    # u is at world position: (i*dx, (j+0.5)*dx, (k+0.5)*dx)
    px = float(i) * grid.dx
    py = (float(j) + 0.5) * grid.dx
    pz = (float(k) + 0.5) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py, pz)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt
    pz_back = pz - vel[2] * dt

    # Sample u at back-traced world position
    grid.u1[i, j, k] = sample_u(grid, px_back, py_back, pz_back)


@wp.kernel
def advect_v(grid: MACGrid3D, dt: float):
    i, j, k = wp.tid()

    # v field size: (nx, ny+1, nz)
    if i >= grid.nx or j >= grid.ny + 1 or k >= grid.nz:
        return

    # v is at world position: ((i+0.5)*dx, j*dx, (k+0.5)*dx)
    px = (float(i) + 0.5) * grid.dx
    py = float(j) * grid.dx
    pz = (float(k) + 0.5) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py, pz)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt
    pz_back = pz - vel[2] * dt

    # Sample v at back-traced world position
    grid.v1[i, j, k] = sample_v(grid, px_back, py_back, pz_back)


@wp.kernel
def advect_w(grid: MACGrid3D, dt: float):
    i, j, k = wp.tid()

    # w field size: (nx, ny, nz+1)
    if i >= grid.nx or j >= grid.ny or k >= grid.nz + 1:
        return

    # w is at world position: ((i+0.5)*dx, (j+0.5)*dx, k*dx)
    px = (float(i) + 0.5) * grid.dx
    py = (float(j) + 0.5) * grid.dx
    pz = float(k) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py, pz)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt
    pz_back = pz - vel[2] * dt

    # Sample w at back-traced world position
    grid.w1[i, j, k] = sample_w(grid, px_back, py_back, pz_back)


@wp.kernel
def advect_scalar(grid: MACGrid3D, field_in: wp.array3d(dtype=float), field_out: wp.array3d(dtype=float), dt: float):
    i, j, k = wp.tid()

    # scalar field size: (nx, ny, nz)
    if i >= grid.nx or j >= grid.ny or k >= grid.nz:
        return

    # Cell center in world position: ((i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx)
    px = (float(i) + 0.5) * grid.dx
    py = (float(j) + 0.5) * grid.dx
    pz = (float(k) + 0.5) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py, pz)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt
    pz_back = pz - vel[2] * dt

    # Sample scalar at back-traced world position
    field_out[i, j, k] = sample_scalar(grid, field_in, px_back, py_back, pz_back)


#######################################################################
# Projection Kernels
#######################################################################

@wp.kernel
def compute_divergence_kernel(grid: MACGrid3D):
    i, j, k = wp.tid()
    if i >= grid.nx or j >= grid.ny or k >= grid.nz:
        return

    # Compute div(u)
    grid.div[i, j, k] = compute_divergence(grid, i, j, k)


@wp.kernel
def pressure_solve_jacobi_kernel(grid: MACGrid3D, p_in: wp.array3d(dtype=float), p_out: wp.array3d(dtype=float), dt: float, rho: float):
    i, j, k = wp.tid()
    if i >= grid.nx or j >= grid.ny or k >= grid.nz:
        return

    # Get sum of neighbors
    neighbor_sum_p = compute_neighbor_pressure(grid, p_in, i, j, k)

    # Pressure Poisson equation: nabla^2 p = (rho/dt) * nabla . u
    # Discretization: (sum_neighbors - 6p) / dx^2 = (rho/dt) * nabla . u
    # Rearranging: p = (sum_neighbors - (rho/dt) * nabla . u * dx^2) / 6

    div = grid.div[i, j, k]
    rhs = div * (rho / dt) * (grid.dx * grid.dx)
    p_out[i, j, k] = (neighbor_sum_p - rhs) / 6.0


@wp.kernel
def projection_kernel(grid: MACGrid3D, dt: float, rho: float):
    # Update velocity by subtracting pressure gradient
    # u_new = u_old - (dt/rho) * grad(p)

    i, j, k = wp.tid()

    # Handle u (nx+1, ny, nz)
    if i < grid.nx + 1 and j < grid.ny and k < grid.nz:
        p_curr = lookup_float(grid.p0, i, j, k)
        p_prev = lookup_float(grid.p0, i-1, j, k)

        grad_p = (p_curr - p_prev) / grid.dx
        grid.u0[i, j, k] = grid.u0[i, j, k] - (dt / rho) * grad_p

    # Handle v (nx, ny+1, nz)
    if i < grid.nx and j < grid.ny + 1 and k < grid.nz:
        p_curr = lookup_float(grid.p0, i, j, k)
        p_prev = lookup_float(grid.p0, i, j-1, k)

        grad_p = (p_curr - p_prev) / grid.dx
        grid.v0[i, j, k] = grid.v0[i, j, k] - (dt / rho) * grad_p

    # Handle w (nx, ny, nz+1)
    if i < grid.nx and j < grid.ny and k < grid.nz + 1:
        p_curr = lookup_float(grid.p0, i, j, k)
        p_prev = lookup_float(grid.p0, i, j, k-1)

        grad_p = (p_curr - p_prev) / grid.dx
        grid.w0[i, j, k] = grid.w0[i, j, k] - (dt / rho) * grad_p


#######################################################################
# Boundary Condition Kernels
#######################################################################

@wp.kernel
def apply_velocity_bc_kernel(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    bc_left: int,
    bc_right: int,
    bc_top: int,
    bc_bottom: int,
    bc_front: int,
    bc_back: int
):
    """
    Apply boundary conditions for velocity.
    BC types: 0=neumann (free-slip), 1=dirichlet (no-slip), 2=open, 3=periodic
    """
    i, j, k = wp.tid()

    # u boundaries (u has shape nx+1, ny, nz)
    if j < ny and k < nz:
        # Left boundary (x=0)
        if i == 0:
            if bc_left == 0 or bc_left == 1:
                u[0, j, k] = 0.0

        # Right boundary (x=nx)
        if i == nx:
            if bc_right == 0 or bc_right == 1:
                u[nx, j, k] = 0.0
            elif bc_right == 2:  # open: extrapolate
                u[nx, j, k] = u[nx - 1, j, k]

    # v boundaries (v has shape nx, ny+1, nz)
    if i < nx and k < nz:
        # Bottom boundary (y=0)
        if j == 0:
            if bc_bottom == 0 or bc_bottom == 1:
                v[i, 0, k] = 0.0

        # Top boundary (y=ny)
        if j == ny:
            if bc_top == 0 or bc_top == 1:
                v[i, ny, k] = 0.0
            elif bc_top == 2:  # open: extrapolate
                v[i, ny, k] = v[i, ny - 1, k]

    # w boundaries (w has shape nx, ny, nz+1)
    if i < nx and j < ny:
        # Front boundary (z=0)
        if k == 0:
            if bc_front == 0 or bc_front == 1:
                w[i, j, 0] = 0.0

        # Back boundary (z=nz)
        if k == nz:
            if bc_back == 0 or bc_back == 1:
                w[i, j, nz] = 0.0
            elif bc_back == 2:  # open: extrapolate
                w[i, j, nz] = w[i, j, nz - 1]


#######################################################################
# StableFluidSolver3D
#######################################################################

class StableFluidSolver3D(Solver):
    def __init__(
        self,
        grid: MACGrid3D,
        dt: float,
        rho_0: float,
        p_iter: int = 100,
        emitter_data: "EmitterData3D" = None,
        mask_data: "MaskData3D" = None,
        bc_flags: dict = None,
        **kwargs
    ):
        """
        Initialize 3D Stable Fluid Solver.

        Args:
            grid: MAC grid structure
            dt: Time step size
            rho_0: Fluid density
            p_iter: Number of pressure solver iterations
            emitter_data: GPU arrays for emitters (prepared by SimulationController)
            mask_data: GPU arrays for masks (prepared by SimulationController)
            bc_flags: Boundary condition flags (prepared by SimulationController)
        """
        self.grid = grid
        self.dt = dt
        self.rho_0 = rho_0
        self.p_iter = p_iter

        # Store GPU data from controller
        self.emitter_data = emitter_data
        self.mask_data = mask_data

        # Boundary condition flags (default to neumann=0)
        self.bc_flags = bc_flags or {
            "left": 0, "right": 0, "top": 0, "bottom": 0, "front": 0, "back": 0
        }

    def step(self):
        """Execute one simulation time step."""
        bc_dim = (self.grid.nx + 1, self.grid.ny + 1, self.grid.nz + 1)

        with wp.ScopedTimer("StableFluid Step", synchronize=True):
            # 1. External Forces - Apply emitters
            if self.emitter_data and self.emitter_data.count > 0:
                wp.launch(
                    kernel=apply_emitters_kernel,
                    dim=self.grid.smoke0.shape,
                    inputs=[
                        self.grid.smoke0,
                        self.grid.u0,
                        self.grid.v0,
                        self.grid.w0,
                        self.emitter_data.centers,
                        self.emitter_data.radii,
                        self.emitter_data.heights,
                        self.emitter_data.velocities,
                        self.emitter_data.smoke_amounts,
                        self.emitter_data.count,
                        self.grid.dx,
                        self.dt
                    ]
                )

            # 2. Apply obstacle masks
            if self.mask_data and self.mask_data.count > 0:
                wp.launch(
                    kernel=apply_masks_kernel,
                    dim=bc_dim,
                    inputs=[
                        self.grid.u0,
                        self.grid.v0,
                        self.grid.w0,
                        self.mask_data.centers,
                        self.mask_data.radii,
                        self.mask_data.count,
                        self.grid.nx,
                        self.grid.ny,
                        self.grid.nz,
                        self.grid.dx
                    ]
                )

            # 3. Apply Velocity BC after external forces
            wp.launch(
                kernel=apply_velocity_bc_kernel,
                dim=bc_dim,
                inputs=[
                    self.grid.u0, self.grid.v0, self.grid.w0,
                    self.grid.nx, self.grid.ny, self.grid.nz,
                    self.bc_flags["left"],
                    self.bc_flags["right"],
                    self.bc_flags["top"],
                    self.bc_flags["bottom"],
                    self.bc_flags["front"],
                    self.bc_flags["back"]
                ]
            )

            # 4. Advection
            # 4-1. Advect Velocity (write to u1, v1, w1)
            wp.launch(kernel=advect_u, dim=self.grid.u0.shape, inputs=[self.grid, self.dt])
            wp.launch(kernel=advect_v, dim=self.grid.v0.shape, inputs=[self.grid, self.dt])
            wp.launch(kernel=advect_w, dim=self.grid.w0.shape, inputs=[self.grid, self.dt])
            # 4-2. Advect Scalars (Smoke, write to smoke1)
            wp.launch(kernel=advect_scalar, dim=self.grid.smoke0.shape, inputs=[self.grid, self.grid.smoke0, self.grid.smoke1, self.dt])

            # Swap buffers after advection
            (self.grid.u0, self.grid.u1) = (self.grid.u1, self.grid.u0)
            (self.grid.v0, self.grid.v1) = (self.grid.v1, self.grid.v0)
            (self.grid.w0, self.grid.w1) = (self.grid.w1, self.grid.w0)
            (self.grid.smoke0, self.grid.smoke1) = (self.grid.smoke1, self.grid.smoke0)

            # Apply Velocity BC after advection
            wp.launch(
                kernel=apply_velocity_bc_kernel,
                dim=bc_dim,
                inputs=[
                    self.grid.u0, self.grid.v0, self.grid.w0,
                    self.grid.nx, self.grid.ny, self.grid.nz,
                    self.bc_flags["left"],
                    self.bc_flags["right"],
                    self.bc_flags["top"],
                    self.bc_flags["bottom"],
                    self.bc_flags["front"],
                    self.bc_flags["back"]
                ]
            )

            # 5. Projection
            # 5-1. Compute Divergence
            wp.launch(kernel=compute_divergence_kernel, dim=self.grid.div.shape, inputs=[self.grid])

            # 5-2. Pressure Solve (Jacobi)
            self.grid.p0.zero_()
            self.grid.p1.zero_()
            for _ in range(self.p_iter):
                wp.launch(kernel=pressure_solve_jacobi_kernel, dim=self.grid.p0.shape, inputs=[self.grid, self.grid.p0, self.grid.p1, self.dt, self.rho_0])
                (self.grid.p0, self.grid.p1) = (self.grid.p1, self.grid.p0)

            # 5-3. Subtract Gradient
            wp.launch(kernel=projection_kernel, dim=bc_dim, inputs=[self.grid, self.dt, self.rho_0])

            # 5-4. Apply Velocity BC after projection
            wp.launch(
                kernel=apply_velocity_bc_kernel,
                dim=bc_dim,
                inputs=[
                    self.grid.u0, self.grid.v0, self.grid.w0,
                    self.grid.nx, self.grid.ny, self.grid.nz,
                    self.bc_flags["left"],
                    self.bc_flags["right"],
                    self.bc_flags["top"],
                    self.bc_flags["bottom"],
                    self.bc_flags["front"],
                    self.bc_flags["back"]
                ]
            )

            # 6. Apply masks again after projection
            if self.mask_data and self.mask_data.count > 0:
                wp.launch(
                    kernel=apply_masks_kernel,
                    dim=bc_dim,
                    inputs=[
                        self.grid.u0,
                        self.grid.v0,
                        self.grid.w0,
                        self.mask_data.centers,
                        self.mask_data.radii,
                        self.mask_data.count,
                        self.grid.nx,
                        self.grid.ny,
                        self.grid.nz,
                        self.grid.dx
                    ]
                )
