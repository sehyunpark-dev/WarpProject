import warp as wp
from solvers.base_solver import Solver
from core.mac_grid_2d import MACGrid2D, lookup_float, sample_float, sample_scalar, sample_velocity, sample_u, sample_v, compute_divergence, compute_neighbor_pressure

#######################################################################
# External Force Kernels

@wp.kernel
def source_smoke_and_velocity_kernel(
    smoke: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    center: wp.vec2,
    radius: float,
    smoke_amount: float,
    source_speed: float,
    dx: float,
    dt: float
):
    """
    Add smoke source and apply body force (velocity injection) only within the source region.
    This kernel handles both smoke injection and velocity forcing in a single pass.
    In 2D, the source is a circle (disk) instead of a cylinder.
    """
    i, j = wp.tid()

    nx = smoke.shape[0]
    ny = smoke.shape[1]

    if i >= nx or j >= ny:
        return

    # Cell center in world coordinates
    px = (float(i) + 0.5) * dx
    py = (float(j) + 0.5) * dx

    # Distance from center (circular source in 2D)
    dist = wp.sqrt((px - center[0]) * (px - center[0]) + (py - center[1]) * (py - center[1]))

    # Check if within circle
    if dist < radius:
        # Inject smoke
        smoke[i, j] = smoke_amount

        # Apply body force to v-faces adjacent to this cell
        # v[i, j] is at the bottom face of cell (i, j)
        # v[i, j+1] is at the top face of cell (i, j)
        # We add velocity to both faces to inject upward momentum
        v[i, j] = v[i, j] + source_speed * dt
        v[i, j + 1] = v[i, j + 1] + source_speed * dt

#######################################################################

#######################################################################
# Advection Kernels

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

#######################################################################
# Projection Kernel

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

    # Pressure Poisson equation: ∇²p = (ρ/dt) * ∇·u
    # Discretization: (sum_neighbors - 4p) / dx² = (ρ/dt) * ∇·u
    # Rearranging: p = (sum_neighbors - (rho/dt) * ∇·u * dx²) / 4

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

@wp.kernel
def apply_velocity_bc_kernel(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), nx: int, ny: int):
    """
    Apply Neumann BC for velocity:
    - No-penetration: Normal velocity = 0 at domain boundaries
    - Free-slip: Tangential velocity unchanged (du/dn = 0)
    """
    i, j = wp.tid()

    # u boundaries (u has shape nx+1, ny)
    # u[0, j] = 0 and u[nx, j] = 0 (no flow through x boundaries)
    if j < ny:
        if i == 0:
            u[0, j] = 0.0
        if i == nx:
            u[nx, j] = 0.0

    # v boundaries (v has shape nx, ny+1)
    # v[i, 0] = 0 and v[i, ny] = 0 (no flow through y boundaries)
    if i < nx:
        if j == 0:
            v[i, 0] = 0.0
        if j == ny:
            v[i, ny] = 0.0

#######################################################################

class StableFluidSolver2D(Solver):
    def __init__(self, grid: MACGrid2D, dt: float, rho_0: float, p_iter=100, source_speed=100.0, **kwargs):
        self.grid = grid
        self.dt = dt
        self.rho_0 = rho_0
        self.p_iter = p_iter
        self.source_speed = source_speed  # Velocity injection at source (m/s)

        # Source settings (world coordinates)
        # Center of the circular source
        self.source_center = wp.vec2(0.5, 0.05)  # (x, y) in world units
        self.source_radius = 0.05   # Radius (world units)
        self.source_amount = 1.0

    def step(self):
        bc_dim = (self.grid.nx + 1, self.grid.ny + 1)

        with wp.ScopedTimer("StableFluid Step", synchronize=True):
            # 1. External Forces
            # Add smoke source and inject velocity (body force) only within the source region
            wp.launch(
                kernel=source_smoke_and_velocity_kernel,
                dim=self.grid.smoke0.shape,
                inputs=[
                    self.grid.smoke0,
                    self.grid.v0,
                    self.source_center,
                    self.source_radius,
                    self.source_amount,
                    self.source_speed,
                    self.grid.dx,
                    self.dt
                ]
            )
            # Apply Velocity BC after external forces
            wp.launch(kernel=apply_velocity_bc_kernel, dim=bc_dim, inputs=[self.grid.u0, self.grid.v0, self.grid.nx, self.grid.ny])

            # 2. Advection
            # 2-1. Advect Velocity (write to u1, v1)
            wp.launch(kernel=advect_u, dim=self.grid.u0.shape, inputs=[self.grid, self.dt])
            wp.launch(kernel=advect_v, dim=self.grid.v0.shape, inputs=[self.grid, self.dt])
            # 2-2. Advect Scalars (Smoke, write to smoke1)
            wp.launch(kernel=advect_scalar, dim=self.grid.smoke0.shape, inputs=[self.grid, self.grid.smoke0, self.grid.smoke1, self.dt])

            # Swap buffers after advection
            (self.grid.u0, self.grid.u1) = (self.grid.u1, self.grid.u0)
            (self.grid.v0, self.grid.v1) = (self.grid.v1, self.grid.v0)
            (self.grid.smoke0, self.grid.smoke1) = (self.grid.smoke1, self.grid.smoke0)

            # Apply Velocity BC after advection
            wp.launch(kernel=apply_velocity_bc_kernel, dim=bc_dim, inputs=[self.grid.u0, self.grid.v0, self.grid.nx, self.grid.ny])

            # 3. Projection
            # 3-1. Compute Divergence
            wp.launch(kernel=compute_divergence_kernel, dim=self.grid.div.shape, inputs=[self.grid])

            # 3-2. Pressure Solve (Jacobi)
            self.grid.p0.zero_()
            self.grid.p1.zero_()
            for _ in range(self.p_iter):
                # Write to p1
                wp.launch(kernel=pressure_solve_jacobi_kernel, dim=self.grid.p0.shape, inputs=[self.grid, self.grid.p0, self.grid.p1, self.dt, self.rho_0])
                # Swap pressure buffers
                (self.grid.p0, self.grid.p1) = (self.grid.p1, self.grid.p0)

            # 3-3. Subtract Gradient
            wp.launch(kernel=projection_kernel, dim=bc_dim, inputs=[self.grid, self.dt, self.rho_0])

            # 3-4. Apply Velocity BC after projection
            wp.launch(kernel=apply_velocity_bc_kernel, dim=bc_dim, inputs=[self.grid.u0, self.grid.v0, self.grid.nx, self.grid.ny])
