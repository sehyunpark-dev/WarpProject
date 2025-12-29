import warp as wp
from typing import TYPE_CHECKING

from solvers.base_solver import Solver
from core.dim2d.mac_grid_2d import MACGrid2D, lookup_float, sample_float, sample_scalar, sample_velocity, sample_u, sample_v, compute_divergence, compute_neighbor_pressure

if TYPE_CHECKING:
    from core.dim2d.simulation_2d import EmitterData2D, MaskData2D


#######################################################################
# External Force Kernels
#######################################################################

@wp.kernel
def apply_emitters_kernel(
    smoke: wp.array2d(dtype=float),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    emitter_centers: wp.array1d(dtype=wp.vec2),
    emitter_radii: wp.array1d(dtype=float),
    emitter_velocities: wp.array1d(dtype=wp.vec2),
    emitter_smoke_amounts: wp.array1d(dtype=float),
    num_emitters: int,
    dx: float,
    dt: float
):
    """
    Apply multiple emitters: inject smoke and velocity.
    Each emitter is a circular region in 2D.
    """
    i, j = wp.tid()

    nx = smoke.shape[0]
    ny = smoke.shape[1]

    if i >= nx or j >= ny:
        return

    # Cell center in world coordinates
    px = (float(i) + 0.5) * dx
    py = (float(j) + 0.5) * dx

    # Check all emitters
    for e in range(num_emitters):
        center = emitter_centers[e]
        radius = emitter_radii[e]
        vel = emitter_velocities[e]
        amount = emitter_smoke_amounts[e]

        # Distance from emitter center (circular source)
        dist = wp.sqrt((px - center[0]) * (px - center[0]) + (py - center[1]) * (py - center[1]))

        if dist < radius:
            # Inject smoke
            smoke[i, j] = amount

            # Apply velocity to adjacent faces
            # u-faces (horizontal velocity)
            if wp.abs(vel[0]) > 0.0:
                u[i, j] = vel[0]
                if i + 1 < nx + 1:
                    u[i + 1, j] = vel[0]

            # v-faces (vertical velocity)
            if wp.abs(vel[1]) > 0.0:
                v[i, j] = vel[1]
                if j + 1 < ny + 1:
                    v[i, j + 1] = vel[1]


@wp.kernel
def apply_masks_kernel(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    mask_centers: wp.array1d(dtype=wp.vec2),
    mask_radii: wp.array1d(dtype=float),
    num_masks: int,
    nx: int,
    ny: int,
    dx: float
):
    """
    Apply obstacle masks: set velocity to zero inside solid regions.
    """
    i, j = wp.tid()

    # Process u-faces (shape: nx+1, ny)
    if i <= nx and j < ny:
        px_u = float(i) * dx
        py_u = (float(j) + 0.5) * dx

        for m in range(num_masks):
            center = mask_centers[m]
            radius = mask_radii[m]
            dist = wp.sqrt((px_u - center[0]) * (px_u - center[0]) + (py_u - center[1]) * (py_u - center[1]))
            if dist < radius:
                u[i, j] = 0.0

    # Process v-faces (shape: nx, ny+1)
    if i < nx and j <= ny:
        px_v = (float(i) + 0.5) * dx
        py_v = float(j) * dx

        for m in range(num_masks):
            center = mask_centers[m]
            radius = mask_radii[m]
            dist = wp.sqrt((px_v - center[0]) * (px_v - center[0]) + (py_v - center[1]) * (py_v - center[1]))
            if dist < radius:
                v[i, j] = 0.0


#######################################################################
# Advection Kernels
#######################################################################

@wp.kernel
def advect_u(grid: MACGrid2D, dt: float):
    i, j = wp.tid()

    # u field size: (nx+1, ny)
    if i >= grid.nx + 1 or j >= grid.ny:
        return

    # u is at world position: (i*dx, (j+0.5)*dx)
    px = float(i) * grid.dx
    py = (float(j) + 0.5) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt

    # Sample u at back-traced world position
    grid.u1[i, j] = sample_u(grid, px_back, py_back)


@wp.kernel
def advect_v(grid: MACGrid2D, dt: float):
    i, j = wp.tid()

    # v field size: (nx, ny+1)
    if i >= grid.nx or j >= grid.ny + 1:
        return

    # v is at world position: ((i+0.5)*dx, j*dx)
    px = (float(i) + 0.5) * grid.dx
    py = float(j) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt

    # Sample v at back-traced world position
    grid.v1[i, j] = sample_v(grid, px_back, py_back)


@wp.kernel
def advect_scalar(grid: MACGrid2D, field_in: wp.array2d(dtype=float), field_out: wp.array2d(dtype=float), dt: float):
    i, j = wp.tid()

    # scalar field size: (nx, ny)
    if i >= grid.nx or j >= grid.ny:
        return

    # Cell center in world position: ((i+0.5)*dx, (j+0.5)*dx)
    px = (float(i) + 0.5) * grid.dx
    py = (float(j) + 0.5) * grid.dx

    # Get velocity at current world position
    vel = sample_velocity(grid, px, py)

    # Back-trace in world coordinates
    px_back = px - vel[0] * dt
    py_back = py - vel[1] * dt

    # Sample scalar at back-traced world position
    field_out[i, j] = sample_scalar(grid, field_in, px_back, py_back)


#######################################################################
# Projection Kernels
#######################################################################

@wp.kernel
def compute_divergence_kernel(grid: MACGrid2D):
    i, j = wp.tid()
    if i >= grid.nx or j >= grid.ny:
        return

    # Compute div(u)
    grid.div[i, j] = compute_divergence(grid, i, j)


@wp.kernel
def pressure_solve_jacobi_kernel(grid: MACGrid2D, p_in: wp.array2d(dtype=float), p_out: wp.array2d(dtype=float), dt: float, rho: float):
    i, j = wp.tid()
    if i >= grid.nx or j >= grid.ny:
        return

    # Get sum of neighbors
    neighbor_sum_p = compute_neighbor_pressure(grid, p_in, i, j)

    # Pressure Poisson equation: nabla^2 p = (rho/dt) * nabla . u
    # Discretization: (sum_neighbors - 4p) / dx^2 = (rho/dt) * nabla . u
    # Rearranging: p = (sum_neighbors - (rho/dt) * nabla . u * dx^2) / 4

    div = grid.div[i, j]
    rhs = div * (rho / dt) * (grid.dx * grid.dx)
    p_out[i, j] = (neighbor_sum_p - rhs) / 4.0


@wp.kernel
def projection_kernel(grid: MACGrid2D, dt: float, rho: float):
    # Update velocity by subtracting pressure gradient
    # u_new = u_old - (dt/rho) * grad(p)

    i, j = wp.tid()

    # Handle u (nx+1, ny)
    if i < grid.nx + 1 and j < grid.ny:
        # Gradient at face (i, j)
        # p is at cell centers. u is at face between i-1 and i.
        # grad_p_x = (p[i] - p[i-1]) / dx

        p_curr = lookup_float(grid.p0, i, j)
        p_prev = lookup_float(grid.p0, i-1, j)

        grad_p = (p_curr - p_prev) / grid.dx
        grid.u0[i, j] = grid.u0[i, j] - (dt / rho) * grad_p

    # Handle v (nx, ny+1)
    if i < grid.nx and j < grid.ny + 1:
        p_curr = lookup_float(grid.p0, i, j)
        p_prev = lookup_float(grid.p0, i, j-1)

        grad_p = (p_curr - p_prev) / grid.dx
        grid.v0[i, j] = grid.v0[i, j] - (dt / rho) * grad_p


#######################################################################
# Boundary Condition Kernels
#######################################################################

@wp.kernel
def apply_velocity_bc_kernel(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    nx: int,
    ny: int,
    bc_left: int,
    bc_right: int,
    bc_top: int,
    bc_bottom: int
):
    """
    Apply boundary conditions for velocity.
    BC types: 0=neumann (free-slip), 1=dirichlet (no-slip), 2=open, 3=periodic
    """
    i, j = wp.tid()

    # u boundaries (u has shape nx+1, ny)
    if j < ny:
        # Left boundary (x=0)
        if i == 0:
            if bc_left == 0 or bc_left == 1:  # neumann or dirichlet: no penetration
                u[0, j] = 0.0
            # open (bc_left == 2): velocity unchanged (let emitter control)

        # Right boundary (x=nx)
        if i == nx:
            if bc_right == 0 or bc_right == 1:  # neumann or dirichlet
                u[nx, j] = 0.0
            elif bc_right == 2:  # open: extrapolate
                u[nx, j] = u[nx - 1, j]

    # v boundaries (v has shape nx, ny+1)
    if i < nx:
        # Bottom boundary (y=0)
        if j == 0:
            if bc_bottom == 0 or bc_bottom == 1:  # neumann or dirichlet
                v[i, 0] = 0.0
            # open: velocity unchanged

        # Top boundary (y=ny)
        if j == ny:
            if bc_top == 0 or bc_top == 1:  # neumann or dirichlet
                v[i, ny] = 0.0
            elif bc_top == 2:  # open: extrapolate
                v[i, ny] = v[i, ny - 1]


#######################################################################
# StableFluidSolver2D
#######################################################################

class StableFluidSolver2D(Solver):
    def __init__(
        self,
        grid: MACGrid2D,
        dt: float,
        rho_0: float,
        p_iter: int = 100,
        emitter_data: "EmitterData2D" = None,
        mask_data: "MaskData2D" = None,
        bc_flags: dict = None,
        **kwargs
    ):
        """
        Initialize 2D Stable Fluid Solver.

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
            "left": 0, "right": 0, "top": 0, "bottom": 0
        }

    def step(self):
        """Execute one simulation time step."""
        bc_dim = (self.grid.nx + 1, self.grid.ny + 1)

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
                        self.emitter_data.centers,
                        self.emitter_data.radii,
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
                        self.mask_data.centers,
                        self.mask_data.radii,
                        self.mask_data.count,
                        self.grid.nx,
                        self.grid.ny,
                        self.grid.dx
                    ]
                )

            # 3. Apply Velocity BC after external forces
            wp.launch(
                kernel=apply_velocity_bc_kernel,
                dim=bc_dim,
                inputs=[
                    self.grid.u0, self.grid.v0,
                    self.grid.nx, self.grid.ny,
                    self.bc_flags["left"],
                    self.bc_flags["right"],
                    self.bc_flags["top"],
                    self.bc_flags["bottom"]
                ]
            )

            # 4. Advection
            # 4-1. Advect Velocity (write to u1, v1)
            wp.launch(kernel=advect_u, dim=self.grid.u0.shape, inputs=[self.grid, self.dt])
            wp.launch(kernel=advect_v, dim=self.grid.v0.shape, inputs=[self.grid, self.dt])
            # 4-2. Advect Scalars (Smoke, write to smoke1)
            wp.launch(kernel=advect_scalar, dim=self.grid.smoke0.shape, inputs=[self.grid, self.grid.smoke0, self.grid.smoke1, self.dt])

            # Swap buffers after advection
            (self.grid.u0, self.grid.u1) = (self.grid.u1, self.grid.u0)
            (self.grid.v0, self.grid.v1) = (self.grid.v1, self.grid.v0)
            (self.grid.smoke0, self.grid.smoke1) = (self.grid.smoke1, self.grid.smoke0)

            # Apply Velocity BC after advection
            wp.launch(
                kernel=apply_velocity_bc_kernel,
                dim=bc_dim,
                inputs=[
                    self.grid.u0, self.grid.v0,
                    self.grid.nx, self.grid.ny,
                    self.bc_flags["left"],
                    self.bc_flags["right"],
                    self.bc_flags["top"],
                    self.bc_flags["bottom"]
                ]
            )

            # 5. Projection
            # 5-1. Compute Divergence
            wp.launch(kernel=compute_divergence_kernel, dim=self.grid.div.shape, inputs=[self.grid])

            # 5-2. Pressure Solve (Jacobi)
            self.grid.p0.zero_()
            self.grid.p1.zero_()
            for _ in range(self.p_iter):
                # Write to p1
                wp.launch(kernel=pressure_solve_jacobi_kernel, dim=self.grid.p0.shape, inputs=[self.grid, self.grid.p0, self.grid.p1, self.dt, self.rho_0])
                # Swap pressure buffers
                (self.grid.p0, self.grid.p1) = (self.grid.p1, self.grid.p0)

            # 5-3. Subtract Gradient
            wp.launch(kernel=projection_kernel, dim=bc_dim, inputs=[self.grid, self.dt, self.rho_0])

            # 5-4. Apply Velocity BC after projection
            wp.launch(
                kernel=apply_velocity_bc_kernel,
                dim=bc_dim,
                inputs=[
                    self.grid.u0, self.grid.v0,
                    self.grid.nx, self.grid.ny,
                    self.bc_flags["left"],
                    self.bc_flags["right"],
                    self.bc_flags["top"],
                    self.bc_flags["bottom"]
                ]
            )

            # 6. Apply masks again after projection (ensure zero velocity in obstacles)
            if self.mask_data and self.mask_data.count > 0:
                wp.launch(
                    kernel=apply_masks_kernel,
                    dim=bc_dim,
                    inputs=[
                        self.grid.u0,
                        self.grid.v0,
                        self.mask_data.centers,
                        self.mask_data.radii,
                        self.mask_data.count,
                        self.grid.nx,
                        self.grid.ny,
                        self.grid.dx
                    ]
                )
