import warp as wp
import numpy as np

@wp.struct
class MACGrid2D:
    nx: int
    ny: int
    dx: float

    # Pressure, Density, Smoke Density, Divergence (Cell Centers)
    p0:     wp.array2d(dtype=float) # size: nx, ny
    p1:     wp.array2d(dtype=float) # size: nx, ny
    smoke0: wp.array2d(dtype=float) # size: nx, ny
    smoke1: wp.array2d(dtype=float) # size: nx, ny
    div:    wp.array2d(dtype=float) # size: nx, ny

    # Velocity (Staggered Faces)
    u0: wp.array2d(dtype=float) # size: nx+1, ny
    u1: wp.array2d(dtype=float) # size: nx+1, ny
    v0: wp.array2d(dtype=float) # size: nx, ny+1
    v1: wp.array2d(dtype=float) # size: nx, ny+1

def create_grid(domain_size=(1.0, 1.0), dx=1.0/128.0, device=None):
    if device is None:
        device = wp.get_device()

    dx = float(dx)
    nx = int(round(domain_size[0] / dx))
    ny = int(round(domain_size[1] / dx))

    grid = MACGrid2D()
    grid.nx = nx
    grid.ny = ny
    grid.dx = dx

    # Cell-centered fields
    shape_cell = (nx, ny)
    grid.p0     = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.p1     = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.smoke0 = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.smoke1 = wp.zeros(shape=shape_cell, dtype=float, device=device)
    grid.div    = wp.zeros(shape=shape_cell, dtype=float, device=device)

    # Face-centered fields
    grid.u0 = wp.zeros(shape=(nx+1, ny), dtype=float, device=device)
    grid.u1 = wp.zeros(shape=(nx+1, ny), dtype=float, device=device)
    grid.v0 = wp.zeros(shape=(nx, ny+1), dtype=float, device=device)
    grid.v1 = wp.zeros(shape=(nx, ny+1), dtype=float, device=device)

    return grid

@wp.func
def pos_to_cell_index(pos: wp.vec2, dx: float, grid_size: wp.vec2i):
    """
    Converts a world position to a cell index given dx and grid sizes.
    """
    # Calculate index
    i = int(pos[0] / dx)
    j = int(pos[1] / dx)

    # Clamp to valid range [0, N-1]
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0] - 1)
    j = wp.clamp(j, 0, grid_size[1] - 1)

    return wp.vec2i(i, j)

@wp.func
def pos_to_face_index_u(pos: wp.vec2, dx: float, grid_size: wp.vec2i):
    """
    Converts world position to u-face (staggered x) index.
    u is defined at (i, j+0.5)
    """
    # x is face-aligned, y is cell-centered (shifted by 0.5)
    i = int(pos[0] / dx)
    j = int((pos[1] / dx) - 0.5)

    # Clamp: u has nx+1 faces in x-direction
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0])     # [0, Nx]
    j = wp.clamp(j, 0, grid_size[1] - 1) # [0, Ny-1]

    return wp.vec2i(i, j)

@wp.func
def pos_to_face_index_v(pos: wp.vec2, dx: float, grid_size: wp.vec2i):
    """
    Converts world position to v-face (staggered y) index.
    v is defined at (i+0.5, j)
    """
    # y is face-aligned, x is cell-centered
    i = int((pos[0] / dx) - 0.5)
    j = int(pos[1] / dx)

    # Clamp: v has ny+1 faces in y-direction
    # Neumann boundary conditions assumed
    i = wp.clamp(i, 0, grid_size[0] - 1) # [0, Nx-1]
    j = wp.clamp(j, 0, grid_size[1])     # [0, Ny]

    return wp.vec2i(i, j)

@wp.func
def idx_to_pos_face_u(i: int, j: int, dx: float):
    """
    Converts u-face index to world position.
    u is defined at (i, j+0.5)
    """
    x = float(i) * dx
    y = (float(j) + 0.5) * dx
    return wp.vec2(x, y)

@wp.func
def idx_to_pos_face_v(i: int, j: int, dx: float):
    """
    Converts v-face index to world position.
    v is defined at (i+0.5, j)
    """
    x = (float(i) + 0.5) * dx
    y = float(j) * dx
    return wp.vec2(x, y)

@wp.func
def idx_to_pos_cell(i: int, j: int, dx: float):
    """
    Converts cell index to world position (cell center).
    cell center is defined at (i+0.5, j+0.5)
    """
    x = (float(i) + 0.5) * dx
    y = (float(j) + 0.5) * dx
    return wp.vec2(x, y)

@wp.func
def lookup_float(data: wp.array2d(dtype=float), ix: int, iy: int):
    """
    Lookup value in 2D array indices with clamping.
    """
    nx = data.shape[0]
    ny = data.shape[1]

    ix = wp.clamp(ix, 0, nx - 1)
    iy = wp.clamp(iy, 0, ny - 1)
    return data[ix, iy]

@wp.func
def sample_float(data: wp.array2d(dtype=float), gx: float, gy: float):
    """
    Bilinear sampling of 2D array at fractional indices.
    """
    ix = int(wp.floor(gx))
    iy = int(wp.floor(gy))

    tx = gx - float(ix)
    ty = gy - float(iy)

    # Interpolate along x
    c0 = wp.lerp(lookup_float(data, ix, iy), lookup_float(data, ix+1, iy), tx)
    c1 = wp.lerp(lookup_float(data, ix, iy+1), lookup_float(data, ix+1, iy+1), tx)

    # Interpolate along y
    return wp.lerp(c0, c1, ty)

@wp.func
def sample_u(grid: MACGrid2D, px: float, py: float):
    """
    Sample u velocity at world position (px, py).
    Internally converts to grid index space.
    """
    # Convert world position to grid index
    gx = px / grid.dx
    gy = py / grid.dx
    # u is at (i, j+0.5), so shift y by -0.5
    return sample_float(grid.u0, gx, gy - 0.5)

@wp.func
def sample_v(grid: MACGrid2D, px: float, py: float):
    """
    Sample v velocity at world position (px, py).
    Internally converts to grid index space.
    """
    # Convert world position to grid index
    gx = px / grid.dx
    gy = py / grid.dx
    # v is at (i+0.5, j), so shift x by -0.5
    return sample_float(grid.v0, gx - 0.5, gy)

@wp.func
def sample_velocity(grid: MACGrid2D, px: float, py: float):
    """
    Sample velocity at world position (px, py).
    Returns velocity in world units/s.
    """
    u = sample_u(grid, px, py)
    v = sample_v(grid, px, py)
    return wp.vec2(u, v)

@wp.func
def sample_scalar(grid: MACGrid2D, field: wp.array2d(dtype=float), px: float, py: float):
    """
    Sample scalar field at world position (px, py).
    Cell-centered data: stored at (i,j) representing center at ((i+0.5)*dx, (j+0.5)*dx)
    """
    # Convert world position to grid index, then shift by -0.5 for cell-centered data
    gx = px / grid.dx - 0.5
    gy = py / grid.dx - 0.5
    return sample_float(field, gx, gy)

@wp.func
def compute_divergence(grid: MACGrid2D, i: int, j: int):
    # Divergence = du/dx + dv/dy
    # u is at faces (i, j) and (i+1, j)
    # v is at faces (i, j) and (i, j+1)

    u_left  = grid.u0[i, j]
    u_right = grid.u0[i+1, j]

    v_bottom = grid.v0[i, j]
    v_top    = grid.v0[i, j+1]

    div = ((u_right - u_left) + (v_top - v_bottom)) / grid.dx

    return div

@wp.func
def compute_neighbor_pressure(grid: MACGrid2D, p_in: wp.array2d(dtype=float), i: int, j: int):
    # Jacobi iteration for Poisson equation: nabla^2 p = div * rho / dt
    # Discrete Laplacian: (p_{i+1} + p_{i-1} + p_{j+1} + p_{j-1} - 4p_i) / dx^2
    # Update rule: p_new = (sum(p_neighbors) - div * rho / dt * dx^2) / 4

    # Neighbors (Neumann boundary conditions handled by lookup_float clamping)
    p_left  = lookup_float(p_in, i-1, j)
    p_right = lookup_float(p_in, i+1, j)
    p_down  = lookup_float(p_in, i, j-1)
    p_up    = lookup_float(p_in, i, j+1)

    sum_p = p_left + p_right + p_down + p_up

    return sum_p
