import warp as wp
from core import grid
from solvers.base_solver import Solver
from core.grid import MACGrid3D, lookup_float, sample_float, sample_scalar, sample_velocity, sample_u, sample_v, sample_w, compute_divergence, compute_neighbor_pressure

#######################################################################
# External Force Kernels

@wp.kernel
def apply_buoyancy_kernel(v: wp.array3d(dtype=float), smoke: wp.array3d(dtype=float), buoyancy: float, dt: float):
    i, j, k = wp.tid()
    
    # v shape: nx, ny+1, nz
    nx = v.shape[0]
    ny_face = v.shape[1]
    nz = v.shape[2]
    
    if i >= nx or j >= ny_face or k >= nz:
        return
        
    # v[i, j, k] is at face j. It sits between cell j-1 and cell j.
    # We average smoke density from j-1 and j.
    
    # smoke is (nx, ny, nz)
    # Clamp indices to valid cell range [0, ny-1]
    idx_prev = wp.clamp(j-1, 0, ny_face-2)
    idx_curr = wp.clamp(j, 0, ny_face-2)
    
    s_prev = smoke[i, idx_prev, k]
    s_curr = smoke[i, idx_curr, k]
    
    s_avg = (s_prev + s_curr) * 0.5
    
    # Apply buoyancy force: F = buoyancy * smoke * up
    # v_new = v_old + F * dt
    v[i, j, k] = v[i, j, k] + buoyancy * s_avg * dt
    
@wp.kernel
def source_smoke_kernel(smoke: wp.array3d(dtype=float), center: wp.vec3, radius: float, amount: float, dt: float):
    i, j, k = wp.tid()
    
    if i >= smoke.shape[0] or j >= smoke.shape[1] or k >= smoke.shape[2]:
        return
        
    # Distance in grid cells
    pos = wp.vec3(float(i), float(j), float(k))
    d = wp.length(pos - center)
    
    if d < radius:
        # Set smoke density to amount (emitter)
        smoke[i, j, k] = amount

#######################################################################

#######################################################################
# Advection Kernels

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

#######################################################################
# Projection Kernel

@wp.kernel
def compute_divergence_kernel(grid: MACGrid3D):
    i, j, k = wp.tid()
    if i >= grid.nx or j >= grid.ny or k >= grid.nz:
        return
    
    # Compute ∇⋅u
    grid.div[i, j, k] = compute_divergence(grid, i, j, k)

@wp.kernel
def pressure_solve_jacobi_kernel(grid: MACGrid3D, p_in: wp.array3d(dtype=float), p_out: wp.array3d(dtype=float), dt: float, rho: float):
    i, j, k = wp.tid()
    if i >= grid.nx or j >= grid.ny or k >= grid.nz:
        return
    
    # Get sum of neighbors
    neighbor_sum_p = compute_neighbor_pressure(grid, p_in, i, j, k)
    
    # RHS = (∇⋅u * ρ/dt) * dx^2
    # ∇⋅u is already computed
    div = grid.div[i, j, k]
    
    # p_new = (neighbor_sum_p - RHS) / 6.0
    # ∇^2 p = ∇⋅u * ρ/dt
    # (sum - 6p)/dx^2 = ∇⋅u * ρ/dt
    # sum - 6p = div * rho/dt * dx^2
    # 6p = sum - div * rho/dt * dx^2
    
    rhs = div * (grid.dx * grid.dx)
    p_out[i, j, k] = (neighbor_sum_p - rhs) / 6.0

@wp.kernel
def projection_kernel(grid: MACGrid3D, dt: float, rho: float):
    # Update velocity by subtracting pressure gradient
    # u_new = u_old - (dt/ρ) * ∇p
    
    i, j, k = wp.tid()
    
    # Handle u (nx+1, ny, nz)
    if i < grid.nx + 1 and j < grid.ny and k < grid.nz:
        # Gradient at face (i, j, k)
        # p is at cell centers. u is at face between i-1 and i.
        # grad_p_x = (p[i] - p[i-1]) / dx
        
        # Boundary handling:
        # If i=0, p[i-1] is outside. If i=nx, p[i] is outside.
        # Standard MAC: p[-1] = p[0], p[nx] = p[nx-1] (Neumann)
        # So we can use lookup_float which clamps indices.
        
        p_curr = lookup_float(grid.p0, i, j, k)
        p_prev = lookup_float(grid.p0, i-1, j, k)
        
        grad_p = (p_curr - p_prev) / grid.dx
        grid.u0[i, j, k] = grid.u0[i, j, k] - grad_p

    # Handle v (nx, ny+1, nz)
    if i < grid.nx and j < grid.ny + 1 and k < grid.nz:
        p_curr = lookup_float(grid.p0, i, j, k)
        p_prev = lookup_float(grid.p0, i, j-1, k)
        
        grad_p = (p_curr - p_prev) / grid.dx
        grid.v0[i, j, k] = grid.v0[i, j, k] - grad_p

    # Handle w (nx, ny, nz+1)
    if i < grid.nx and j < grid.ny and k < grid.nz + 1:
        p_curr = lookup_float(grid.p0, i, j, k)
        p_prev = lookup_float(grid.p0, i, j, k-1)
        
        grad_p = (p_curr - p_prev) / grid.dx
        grid.w0[i, j, k] = grid.w0[i, j, k] - grad_p

#######################################################################

class StableFluidSolver3D(Solver):
    def __init__(self, grid: MACGrid3D, dt: float, rho_0: float, nu: float, p_iter=50, buoyancy=100.0, **kwargs):
        self.grid = grid
        self.dt = dt
        self.rho_0 = rho_0
        self.nu = nu
        self.p_iter = p_iter
        self.buoyancy = buoyancy
        
        # Source settings
        self.source_center = wp.vec3(float(grid.nx//2), 5.0, float(grid.nz//2))
        self.source_radius = 5.0
        self.source_amount = 1.0

    def step(self):
        # 1. External Forces
        # 1-1. Add Smoke Source
        wp.launch(kernel=source_smoke_kernel, dim=self.grid.smoke0.shape, inputs=[self.grid.smoke0, self.source_center, self.source_radius, self.source_amount, self.dt])
        
        # 1-2. Apply Buoyancy (to v0)
        wp.launch(kernel=apply_buoyancy_kernel, dim=self.grid.v0.shape, inputs=[self.grid.v0, self.grid.smoke0, self.buoyancy, self.dt])
        
        # 2. Advection
        # 2-1. Advect Velocity
        wp.launch(kernel=advect_u, dim=self.grid.u0.shape, inputs=[self.grid, self.dt])
        wp.launch(kernel=advect_v, dim=self.grid.v0.shape, inputs=[self.grid, self.dt])
        wp.launch(kernel=advect_w, dim=self.grid.w0.shape, inputs=[self.grid, self.dt])
        
        # 2-2. Advect Scalars (Density & Smoke)
        wp.launch(kernel=advect_scalar, dim=self.grid.smoke0.shape, inputs=[self.grid, self.grid.smoke0, self.grid.smoke1, self.dt])
        
        # Swap buffers (Advection)
        (self.grid.u0, self.grid.u1) = (self.grid.u1, self.grid.u0)
        (self.grid.v0, self.grid.v1) = (self.grid.v1, self.grid.v0)
        (self.grid.w0, self.grid.w1) = (self.grid.w1, self.grid.w0)
        (self.grid.smoke0, self.grid.smoke1) = (self.grid.smoke1, self.grid.smoke0)
        
        # 3. Projection
        # 3-1. Compute Divergence
        wp.launch(kernel=compute_divergence_kernel, dim=self.grid.div.shape, inputs=[self.grid])
        
        # 3-2. Pressure Solve (Jacobi)
        for _ in range(self.p_iter):
            wp.launch(kernel=pressure_solve_jacobi_kernel, dim=self.grid.p0.shape, inputs=[self.grid, self.grid.p0, self.grid.p1, self.dt, self.rho_0])
            # Swap pressure buffers
            (self.grid.p0, self.grid.p1) = (self.grid.p1, self.grid.p0)
            
        # 3-3. Subtract Gradient
        max_dim = (self.grid.nx + 1, self.grid.ny + 1, self.grid.nz + 1)
        wp.launch(kernel=projection_kernel, dim=max_dim, inputs=[self.grid, self.dt, self.rho_0])
        