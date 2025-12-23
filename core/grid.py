import warp as wp
import numpy as np

@wp.struct
class MACGrid3D:
    nx: int
    ny: int
    nz: int
    dx: float
    
    # Pressure, Density, Smoke Density, Divergence (Cell Centers)
    p0:     wp.array3d(dtype=float) # size: nx, ny, nz
    p1:     wp.array3d(dtype=float) # size: nx, ny, nz
    smoke0: wp.array3d(dtype=float) # size: nx, ny, nz
    smoke1: wp.array3d(dtype=float) # size: nx, ny, nz
    div:    wp.array3d(dtype=float) # size: nx, ny, nz
    
    # Velocity (Staggered Faces)
    u0: wp.array3d(dtype=float) # size: nx+1, ny, nz
    u1: wp.array3d(dtype=float) # size: nx+1, ny, nz
    v0: wp.array3d(dtype=float) # size: nx, ny+1, nz
    v1: wp.array3d(dtype=float) # size: nx, ny+1, nz
    w0: wp.array3d(dtype=float) # size: nx, ny, nz+1
    w1: wp.array3d(dtype=float) # size: nx, ny, nz+1

def create_grid(domain_size=(1.0, 1.0, 1.0), dx=1.0/128.0, device=None):
    if device is None:
        device = wp.get_device()
        
    dx = float(dx)
    nx = int(round(domain_size[0] / dx))
    ny = int(round(domain_size[1] / dx))
    nz = int(round(domain_size[2] / dx))
    
    grid = MACGrid3D()
    grid.nx = nx
    grid.ny = ny
    grid.nz = nz
    grid.dx = dx
    
    # Cell-centered fields
    shape_cell = (nx, ny, nz)
    grid.p0     = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.p1     = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.smoke0 = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.smoke1 = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.div    = wp.zeros(shape=shape_cell, dtype=float, device=device)
    
    # Face-centered fields
    grid.u0 = wp.zeros(shape=(nx+1, ny, nz), dtype=float, device=device)
    grid.u1 = wp.zeros(shape=(nx+1, ny, nz), dtype=float, device=device)
    grid.v0 = wp.zeros(shape=(nx, ny+1, nz), dtype=float, device=device)
    grid.v1 = wp.zeros(shape=(nx, ny+1, nz), dtype=float, device=device)
    grid.w0 = wp.zeros(shape=(nx, ny, nz+1), dtype=float, device=device)
    grid.w1 = wp.zeros(shape=(nx, ny, nz+1), dtype=float, device=device)
    
    return grid

@wp.func
def pos_to_cell_index(pos: wp.vec3, dx: float, grid_size: wp.vec3i):
    """
    Converts a world position to a cell index given dx and grid sizes.
    """
    # Calculate index
    i = int(pos[0] / dx)
    j = int(pos[1] / dx)
    k = int(pos[2] / dx)

    # Clamp to valid range [0, N-1]
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0] - 1)
    j = wp.clamp(j, 0, grid_size[1] - 1)
    k = wp.clamp(k, 0, grid_size[2] - 1)
    
    return wp.vec3i(i, j, k)

@wp.func
def pos_to_face_index_u(pos: wp.vec3, dx: float, grid_size: wp.vec3i):
    """
    Converts world position to u-face (staggered x) index.
    u is defined at (i, j+0.5, k+0.5)
    """
    # x is face-aligned, y and z are cell-centered (shifted by 0.5)
    i = int(pos[0] / dx)
    j = int((pos[1] / dx) - 0.5)
    k = int((pos[2] / dx) - 0.5)

    # Clamp: u has nx+1 faces in x-direction
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0])     # [0, Nx]
    j = wp.clamp(j, 0, grid_size[1] - 1) # [0, Ny-1]
    k = wp.clamp(k, 0, grid_size[2] - 1) # [0, Nz-1]

    return wp.vec3i(i, j, k)

@wp.func
def pos_to_face_index_v(pos: wp.vec3, dx: float, grid_size: wp.vec3i):
    """
    Converts world position to v-face (staggered y) index.
    v is defined at (i+0.5, j, k+0.5)
    """
    # y is face-aligned, x and z are cell-centered
    i = int((pos[0] / dx) - 0.5)
    j = int(pos[1] / dx)
    k = int((pos[2] / dx) - 0.5)

    # Clamp: v has ny+1 faces in y-direction
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0] - 1) # [0, Nx-1]
    j = wp.clamp(j, 0, grid_size[1])     # [0, Ny]
    k = wp.clamp(k, 0, grid_size[2] - 1) # [0, Nz-1]

    return wp.vec3i(i, j, k)

@wp.func
def pos_to_face_index_w(pos: wp.vec3, dx: float, grid_size: wp.vec3i):
    """
    Converts world position to w-face (staggered z) index.
    w is defined at (i+0.5, j+0.5, k)
    """
    # z is face-aligned, x and y are cell-centered
    i = int((pos[0] / dx) - 0.5)
    j = int((pos[1] / dx) - 0.5)
    k = int(pos[2] / dx)

    # Clamp: w has nz+1 faces in z-direction
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0] - 1) # [0, Nx-1]
    j = wp.clamp(j, 0, grid_size[1] - 1) # [0, Ny-1]
    k = wp.clamp(k, 0, grid_size[2])     # [0, Nz]

    return wp.vec3i(i, j, k)

@wp.func
def idx_to_pos_face_u(i: int, j: int, k: int, dx: float):
    """
    Converts u-face index to world position.
    u is defined at (i, j+0.5, k+0.5)
    """
    x = float(i) * dx
    y = (float(j) + 0.5) * dx
    z = (float(k) + 0.5) * dx
    return wp.vec3(x, y, z)

@wp.func
def idx_to_pos_face_v(i: int, j: int, k: int, dx: float):
    """
    Converts v-face index to world position.
    v is defined at (i+0.5, j, k+0.5)
    """
    x = (float(i) + 0.5) * dx
    y = float(j) * dx
    z = (float(k) + 0.5) * dx
    return wp.vec3(x, y, z)

@wp.func
def idx_to_pos_face_w(i: int, j: int, k: int, dx: float):
    """
    Converts w-face index to world position.
    w is defined at (i+0.5, j+0.5, k)
    """
    x = (float(i) + 0.5) * dx
    y = (float(j) + 0.5) * dx
    z = float(k) * dx
    return wp.vec3(x, y, z)

@wp.func
def idx_to_pos_cell(i: int, j: int, k: int, dx: float):
    """
    Converts cell index to world position (cell center).
    cell center is defined at (i+0.5, j+0.5, k+0.5)
    """
    x = (float(i) + 0.5) * dx
    y = (float(j) + 0.5) * dx
    z = (float(k) + 0.5) * dx
    return wp.vec3(x, y, z)

@wp.func
def lookup_float(data: wp.array3d(dtype=float), ix: int, iy: int, iz: int):
    """
    Lookup value in 3D array indices with clamping.
    """
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]
    
    ix = wp.clamp(ix, 0, nx - 1)
    iy = wp.clamp(iy, 0, ny - 1)
    iz = wp.clamp(iz, 0, nz - 1)
    return data[ix, iy, iz]

@wp.func
def sample_float(data: wp.array3d(dtype=float), gx: float, gy: float, gz: float):
    """
    Trilinear sampling of 3D array at fractional indices.
    """
    ix = int(wp.floor(gx))
    iy = int(wp.floor(gy))
    iz = int(wp.floor(gz))

    tx = gx - float(ix)
    ty = gy - float(iy)
    tz = gz - float(iz)
    
    # Interpolate along x
    # z = iz
    c00 = wp.lerp(lookup_float(data, ix, iy, iz), lookup_float(data, ix+1, iy, iz), tx)
    c10 = wp.lerp(lookup_float(data, ix, iy+1, iz), lookup_float(data, ix+1, iy+1, iz), tx)
    
    # z = iz + 1
    c01 = wp.lerp(lookup_float(data, ix, iy, iz+1), lookup_float(data, ix+1, iy, iz+1), tx)
    c11 = wp.lerp(lookup_float(data, ix, iy+1, iz+1), lookup_float(data, ix+1, iy+1, iz+1), tx)

    # Interpolate along y
    c0 = wp.lerp(c00, c10, ty)
    c1 = wp.lerp(c01, c11, ty)

    # Interpolate along z
    return wp.lerp(c0, c1, tz)

@wp.func
def sample_u(grid: MACGrid3D, px: float, py: float, pz: float):
    """
    Sample u velocity at world position (px, py, pz).
    Internally converts to grid index space.
    """
    # Convert world position to grid index
    gx = px / grid.dx
    gy = py / grid.dx
    gz = pz / grid.dx
    # u is at (i, j+0.5, k+0.5), so shift y and z by -0.5
    return sample_float(grid.u0, gx, gy - 0.5, gz - 0.5)

@wp.func
def sample_v(grid: MACGrid3D, px: float, py: float, pz: float):
    """
    Sample v velocity at world position (px, py, pz).
    Internally converts to grid index space.
    """
    # Convert world position to grid index
    gx = px / grid.dx
    gy = py / grid.dx
    gz = pz / grid.dx
    # v is at (i+0.5, j, k+0.5), so shift x and z by -0.5
    return sample_float(grid.v0, gx - 0.5, gy, gz - 0.5)

@wp.func
def sample_w(grid: MACGrid3D, px: float, py: float, pz: float):
    """
    Sample w velocity at world position (px, py, pz).
    Internally converts to grid index space.
    """
    # Convert world position to grid index
    gx = px / grid.dx
    gy = py / grid.dx
    gz = pz / grid.dx
    # w is at (i+0.5, j+0.5, k), so shift x and y by -0.5
    return sample_float(grid.w0, gx - 0.5, gy - 0.5, gz)

@wp.func
def sample_velocity(grid: MACGrid3D, px: float, py: float, pz: float):
    """
    Sample velocity at world position (px, py, pz).
    Returns velocity in world units/s.
    """
    u = sample_u(grid, px, py, pz)
    v = sample_v(grid, px, py, pz)
    w = sample_w(grid, px, py, pz)
    return wp.vec3(u, v, w)

@wp.func
def sample_scalar(grid: MACGrid3D, field: wp.array3d(dtype=float), px: float, py: float, pz: float):
    """
    Sample scalar field at world position (px, py, pz).
    Cell-centered data: stored at (i,j,k) representing center at ((i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx)
    """
    # Convert world position to grid index, then shift by -0.5 for cell-centered data
    gx = px / grid.dx - 0.5
    gy = py / grid.dx - 0.5
    gz = pz / grid.dx - 0.5
    return sample_float(field, gx, gy, gz)

@wp.func
def compute_divergence(grid: MACGrid3D, i: int, j: int, k: int):
    # Divergence = du/dx + dv/dy + dw/dz
    # u is at faces (i, j, k) and (i+1, j, k)
    # v is at faces (i, j, k) and (i, j+1, k)
    # w is at faces (i, j, k) and (i, j, k+1)
    
    u_left  = grid.u0[i, j, k]
    u_right = grid.u0[i+1, j, k]
    
    v_bottom = grid.v0[i, j, k]
    v_top    = grid.v0[i, j+1, k]
    
    w_back  = grid.w0[i, j, k]
    w_front = grid.w0[i, j, k+1]
    
    div = (u_right - u_left) / grid.dx + \
          (v_top - v_bottom) / grid.dx + \
          (w_front - w_back) / grid.dx

    return div

@wp.func
def compute_neighbor_pressure(grid: MACGrid3D, p_in: wp.array3d(dtype=float), i: int, j: int, k: int):
    # Jacobi iteration for Poisson equation: ∇^2 p = div * rho / dt
    # Discrete Laplacian: (p_{i+1} + p_{i-1} + ... - 6p_i) / dx^2
    # Update rule: p_new = (sum(p_neighbors) - div * rho / dt * dx^2) / 6
    
    # Neighbors (Neumann boundary conditions handled by lookup_float clamping)
    p_left  = lookup_float(p_in, i-1, j, k)
    p_right = lookup_float(p_in, i+1, j, k)
    p_down  = lookup_float(p_in, i, j-1, k)
    p_up    = lookup_float(p_in, i, j+1, k)
    p_back  = lookup_float(p_in, i, j, k-1)
    p_front = lookup_float(p_in, i, j, k+1)
    
    sum_p = p_left + p_right + p_down + p_up + p_back + p_front
    
    return sum_p