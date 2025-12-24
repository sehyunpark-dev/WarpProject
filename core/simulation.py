import warp as wp
import numpy as np
from core.grid import MACGrid3D, create_grid
from solvers.base_solver import Solver

class SimulationController3D:
    def __init__(self,
                 solver_type: type[Solver],
                 domain_size=(1.0, 1.0, 1.0),
                 dx=1.0/128.0,
                 dt=0.01,
                 rho_0=1.225,
                 cfl_check=True,
                 p_iter=100):
        self.device = wp.get_device()
        
        self.dt = dt
        self.rho_0 = rho_0
        self.cfl_check = cfl_check
        self.p_iter = p_iter
        
        self.grid = create_grid(domain_size=domain_size, dx=dx, device=self.device)
        self.nx, self.ny, self.nz = self.grid.nx, self.grid.ny, self.grid.nz
        
        self.solver = solver_type(grid=self.grid, dt=self.dt, rho_0=self.rho_0, cfl_check=self.cfl_check, p_iter=self.p_iter)
        
        print(f"Simulation Initialized on {self.device}")
        print(f"Grid Size: ({self.nx}, {self.ny}, {self.nz}), Domain: {domain_size}, dx: {dx}")
        
        # Initialize grid fields
        self.reset()
    
    def step(self):
        """
        Forwards the simulation by one time step.
        """
        self.solver.step()
        
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