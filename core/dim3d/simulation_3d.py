import warp as wp
import numpy as np
import os
from datetime import datetime
from core.dim3d.mac_grid_3d import MACGrid3D, create_grid
from solvers.base_solver import Solver

#######################################################################
# CFL Number Computation Kernel

@wp.kernel
def compute_cfl_kernel(grid: MACGrid3D, dt: float, max_cfl: wp.array1d(dtype=float)):
    """
    Compute velocity magnitude at cell centers for CFL calculation.
    Samples staggered velocity components and computes magnitude.
    """
    idx = wp.tid()

    if idx >= grid.nx * grid.ny * grid.nz:
        return

    # Convert linear index to 3D coordinates
    i = idx // (grid.ny * grid.nz)
    j = (idx % (grid.ny * grid.nz)) // grid.nz
    k = idx % grid.nz
    dx = grid.dx

    # Interpolate velocity components to cell center
    u_center = (grid.u0[i, j, k] + grid.u0[i+1, j, k]) * 0.5
    v_center = (grid.v0[i, j, k] + grid.v0[i, j+1, k]) * 0.5
    w_center = (grid.w0[i, j, k] + grid.w0[i, j, k+1]) * 0.5

    # Compute magnitude
    vel_mag = wp.length(wp.vec3(u_center, v_center, w_center))
    local_cfl = vel_mag * dt / dx

    # Atomic max to find global maximum CFL
    wp.atomic_max(max_cfl, 0, local_cfl)

#######################################################################

class SimulationController3D:
    def __init__(self,
                 solver_type: type[Solver],
                 domain_size=(1.0, 1.0, 1.0),
                 dx=1.0/128.0,
                 dt=0.01,
                 rho_0=1.225,
                 cfl_check=True,
                 export=True,
                 p_iter=100):
        self.device = wp.get_device()

        self.dt = dt
        self.rho_0 = rho_0
        self.cfl_check = cfl_check
        self.export = export
        self.p_iter = p_iter

        self.grid = create_grid(domain_size=domain_size, dx=dx, device=self.device)
        self.nx, self.ny, self.nz = self.grid.nx, self.grid.ny, self.grid.nz

        self.solver = solver_type(grid=self.grid, dt=self.dt, rho_0=self.rho_0, p_iter=self.p_iter)

        # CFL tracking
        self.max_cfl = wp.zeros(1, dtype=float)

        # Numpy export setup
        self.frame_count = 0
        if self.export:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "numpy", timestamp)
            os.makedirs(self.output_dir, exist_ok=True)

        print(f"Simulation Initialized on {self.device}")
        print(f"Grid Size: ({self.nx}, {self.ny}, {self.nz}), Domain: {domain_size}, dx: {dx}")

        # Initialize grid fields
        self.reset()

    def step(self):
        """
        Forwards the simulation by one time step.
        """
        self.solver.step()

        # CFL check
        if self.cfl_check:
            wp.launch(
                kernel=compute_cfl_kernel,
                dim=(self.nx * self.ny * self.nz),
                inputs=[self.grid, self.dt, self.max_cfl]
            )
            current_cfl = self.max_cfl.numpy()[0]
            self.max_cfl.zero_()
            print(f"Max CFL Number: {current_cfl:.4f}")

        # Numpy export
        if self.export:
            self.export_smoke_to_numpy(f"smoke_frame_{self.frame_count:05d}.npy")
            self.frame_count += 1

    def reset(self):
        """
        Resets the simulation grid to initial conditions.
        """
        self.grid.p0.zero_()
        self.grid.p1.zero_()
        self.grid.smoke0.zero_()
        self.grid.smoke1.zero_()
        self.grid.div.zero_()

        self.grid.u0.zero_()
        self.grid.u1.zero_()
        self.grid.v0.zero_()
        self.grid.v1.zero_()
        self.grid.w0.zero_()
        self.grid.w1.zero_()

    def export_smoke_to_numpy(self, filename: str):
        """
        Export smoke density field to numpy .npy file.
        """
        smoke_np = self.grid.smoke0.numpy()
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, smoke_np)
