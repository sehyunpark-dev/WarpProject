"""
Debug script to inspect boundary cell values at (nx/2, ny-1, nz/2).
Prints pressure gradient and velocity components at the top boundary.
"""

import warp as wp
import numpy as np
from core.simulation import SimulationController3D
from solvers.stable_fluid_3d import StableFluidSolver3D

wp.init()


def debug_boundary_values(sim: SimulationController3D, num_steps: int = 10):
    """
    Run simulation and print boundary values at each step.

    Args:
        sim: Simulation controller instance
        num_steps: Number of steps to run
    """
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    dx = sim.grid.dx
    dt = sim.dt
    rho = sim.rho_0

    # Target cell: (nx/2, ny-1, nz/2) - top boundary center
    i = nx // 2
    j = ny - 1  # Top boundary (y_max)
    k = nz // 2

    print("=" * 80)
    print(f"Debug: Boundary Cell at (i={i}, j={j}, k={k})")
    print(f"Grid Size: ({nx}, {ny}, {nz}), dx={dx}, dt={dt}, rho={rho}")
    print(f"World Position: ({(i+0.5)*dx:.4f}, {(j+0.5)*dx:.4f}, {(k+0.5)*dx:.4f})")
    print("=" * 80)

    for step in range(num_steps):
        # Run one simulation step
        sim.solver.step()

        # Synchronize to ensure GPU operations are complete
        wp.synchronize()

        # Get numpy arrays
        p = sim.grid.p0.numpy()
        u = sim.grid.u0.numpy()
        v = sim.grid.v0.numpy()
        w = sim.grid.w0.numpy()
        div = sim.grid.div.numpy()

        # Pressure at target cell and neighbors
        p_center = p[i, j, k]
        p_below = p[i, j-1, k] if j > 0 else 0.0
        p_above = p[i, min(j+1, ny-1), k]  # Clamped (Neumann BC)

        # Pressure gradient at faces
        # grad_p for v[i, j, k] (bottom face of cell)
        grad_p_v_bottom = (p_center - p_below) / dx if j > 0 else 0.0

        # grad_p for v[i, j+1, k] (top face of cell) - this is at boundary
        # At j+1 = ny, this is the domain boundary
        grad_p_v_top = (p_above - p_center) / dx  # p_above is clamped to p[i, ny-1, k]

        # Velocity components at cell faces
        # u faces: u[i, j, k] (left), u[i+1, j, k] (right)
        u_left = u[i, j, k]
        u_right = u[i+1, j, k]

        # v faces: v[i, j, k] (bottom), v[i, j+1, k] (top - boundary)
        v_bottom = v[i, j, k]
        v_top = v[i, j+1, k] if j+1 <= ny else 0.0  # v has shape (nx, ny+1, nz)

        # w faces: w[i, j, k] (back), w[i, j, k+1] (front)
        w_back = w[i, j, k]
        w_front = w[i, j, k+1]

        # Divergence at cell
        div_cell = div[i, j, k]

        # Cell center velocity (interpolated)
        u_center = (u_left + u_right) / 2.0
        v_center = (v_bottom + v_top) / 2.0
        w_center = (w_back + w_front) / 2.0

        print(f"\n--- Step {step + 1} ---")
        print(f"Pressure:")
        print(f"  p[{i},{j},{k}] (center)    = {p_center:+.6e}")
        print(f"  p[{i},{j-1},{k}] (below)   = {p_below:+.6e}")
        print(f"  p[{i},{min(j+1,ny-1)},{k}] (above/clamped) = {p_above:+.6e}")

        print(f"\nPressure Gradient (for projection):")
        print(f"  grad_p at v[{i},{j},{k}] (bottom face) = {grad_p_v_bottom:+.6e}")
        print(f"  grad_p at v[{i},{j+1},{k}] (top face)  = {grad_p_v_top:+.6e}")

        print(f"\nVelocity at faces:")
        print(f"  u[{i},{j},{k}] (left)   = {u_left:+.6e}")
        print(f"  u[{i+1},{j},{k}] (right) = {u_right:+.6e}")
        print(f"  v[{i},{j},{k}] (bottom) = {v_bottom:+.6e}")
        print(f"  v[{i},{j+1},{k}] (top)   = {v_top:+.6e}  <- Boundary (should be 0)")
        print(f"  w[{i},{j},{k}] (back)   = {w_back:+.6e}")
        print(f"  w[{i},{j},{k+1}] (front) = {w_front:+.6e}")

        print(f"\nCell center velocity (interpolated):")
        print(f"  u_center = {u_center:+.6e}")
        print(f"  v_center = {v_center:+.6e}")
        print(f"  w_center = {w_center:+.6e}")

        print(f"\nDivergence:")
        print(f"  div[{i},{j},{k}] = {div_cell:+.6e}")


def debug_all_top_boundary(sim: SimulationController3D, num_steps: int = 5):
    """
    Run simulation and print statistics for all top boundary cells.
    """
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    j = ny - 1  # Top boundary

    print("=" * 80)
    print(f"Debug: All Top Boundary Cells (j={j})")
    print(f"Grid Size: ({nx}, {ny}, {nz})")
    print("=" * 80)

    for step in range(num_steps):
        sim.solver.step()
        wp.synchronize()

        p = sim.grid.p0.numpy()
        v = sim.grid.v0.numpy()

        # Top boundary v values (should all be 0 due to BC)
        v_top = v[:, ny, :]  # v[i, ny, k] for all i, k

        # Pressure at top cells
        p_top = p[:, j, :]

        # v at bottom face of top cells
        v_bottom_of_top = v[:, j, :]

        print(f"\n--- Step {step + 1} ---")
        print(f"v at top boundary (v[:, {ny}, :]):")
        print(f"  min={v_top.min():+.6e}, max={v_top.max():+.6e}, mean={v_top.mean():+.6e}")

        print(f"v at bottom face of top cells (v[:, {j}, :]):")
        print(f"  min={v_bottom_of_top.min():+.6e}, max={v_bottom_of_top.max():+.6e}, mean={v_bottom_of_top.mean():+.6e}")

        print(f"Pressure at top cells (p[:, {j}, :]):")
        print(f"  min={p_top.min():+.6e}, max={p_top.max():+.6e}, mean={p_top.mean():+.6e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug boundary values")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--mode", choices=["single", "all"], default="single",
                        help="single: debug single cell, all: debug all top boundary")
    parser.add_argument("--dx", type=float, default=1.0/64.0, help="Grid spacing")
    args = parser.parse_args()

    # Use smaller grid for faster debugging
    sim = SimulationController3D(
        solver_type=StableFluidSolver3D,
        domain_size=(1.0, 1.0, 1.0),
        dx=args.dx,
        dt=0.005,
        cfl_check=False,
        export=False,
        p_iter=100,
    )

    if args.mode == "single":
        debug_boundary_values(sim, num_steps=args.steps)
    else:
        debug_all_top_boundary(sim, num_steps=args.steps)
