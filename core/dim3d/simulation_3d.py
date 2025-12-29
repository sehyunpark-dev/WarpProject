import warp as wp
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from core.dim3d.mac_grid_3d import MACGrid3D, create_grid
from solvers.base_solver import Solver
from solvers.stable_fluid_3d import apply_initial_velocity_kernel

if TYPE_CHECKING:
    from scene_parser import SimulationConfig, Emitter, Mask, BoundaryCondition, InitialVelocity


#######################################################################
# Data Classes for GPU Data
#######################################################################

@dataclass
class EmitterData3D:
    """Emitter data prepared for GPU kernels"""
    centers: Optional[wp.array] = None       # wp.vec3 array
    radii: Optional[wp.array] = None         # float array (radius in XZ plane)
    heights: Optional[wp.array] = None       # float array (height in Y direction)
    velocities: Optional[wp.array] = None    # wp.vec3 array
    smoke_amounts: Optional[wp.array] = None # float array
    count: int = 0


@dataclass
class MaskData3D:
    """Mask (obstacle) data prepared for GPU kernels"""
    centers: Optional[wp.array] = None  # wp.vec3 array
    radii: Optional[wp.array] = None    # float array
    count: int = 0


@dataclass
class InitialVelocityData3D:
    """Initial velocity region data prepared for GPU kernels"""
    centers: Optional[wp.array] = None      # wp.vec3 array
    half_sizes: Optional[wp.array] = None   # wp.vec3 array (half width, half height, half depth)
    velocities: Optional[wp.array] = None   # wp.vec3 array
    count: int = 0


#######################################################################
# CFL Number Computation Kernel
#######################################################################

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
# SimulationController3D
#######################################################################

class SimulationController3D:
    def __init__(self,
                 solver_type: type[Solver],
                 config: "SimulationConfig"):
        """
        Initialize 3D simulation controller with scene configuration.

        Args:
            solver_type: Solver class to use (e.g., StableFluidSolver3D)
            config: Parsed SimulationConfig from scene JSON file
        """
        self.device = wp.get_device()
        self.config = config

        # Extract settings from config
        scene = config.scene
        solver_cfg = config.solver

        self.dt = solver_cfg.dt
        self.rho_0 = solver_cfg.rho
        self.cfl_check = scene.cfl_check
        self.export = scene.export
        self.p_iter = solver_cfg.p_iter

        # Create grid
        self.grid = create_grid(
            domain_size=scene.domain_size,
            dx=scene.dx,
            device=self.device
        )
        self.nx, self.ny, self.nz = self.grid.nx, self.grid.ny, self.grid.nz

        # Prepare GPU data from config (Controller handles parsing)
        emitter_data = self._prepare_emitters(config.emitters)
        mask_data = self._prepare_masks(config.masks)
        self.initial_velocity_data = self._prepare_initial_velocities(config.initial_velocities)
        bc_flags = self._prepare_bc(scene.bc)

        # Create solver with prepared GPU data
        self.solver = solver_type(
            grid=self.grid,
            dt=self.dt,
            rho_0=self.rho_0,
            p_iter=self.p_iter,
            emitter_data=emitter_data,
            mask_data=mask_data,
            bc_flags=bc_flags
        )

        # CFL tracking
        self.max_cfl = wp.zeros(1, dtype=float)

        # Numpy export setup
        self.frame_count = 0
        if self.export:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "outputs", "numpy", timestamp
            )
            os.makedirs(self.output_dir, exist_ok=True)

        print(f"Simulation Initialized on {self.device}")
        print(f"Grid Size: ({self.nx}, {self.ny}, {self.nz}), Domain: {scene.domain_size}, dx: {scene.dx}")
        print(f"Emitters: {emitter_data.count}, Masks: {mask_data.count}, InitialVelocities: {self.initial_velocity_data.count}")

        # Initialize grid fields
        self.reset()

    def _prepare_emitters(self, emitters: List["Emitter"]) -> EmitterData3D:
        """Convert Emitter list to GPU arrays."""
        n = len(emitters)
        if n == 0:
            return EmitterData3D()

        centers = []
        radii = []
        heights = []
        velocities = []
        smoke_amounts = []

        for e in emitters:
            # Center position (x, y, z)
            centers.append((e.center[0], e.center[1], e.center[2]))

            # Handle different shapes
            if e.shape == "sphere":
                radii.append(e.params.get("radius", 0.05))
                heights.append(e.params.get("radius", 0.05) * 2.0)  # diameter as height
            elif e.shape == "cylinder":
                radii.append(e.params.get("radius", 0.05))
                heights.append(e.params.get("height", 0.05))
            elif e.shape == "box":
                w = e.params.get("width", 0.1)
                h = e.params.get("height", 0.1)
                d = e.params.get("depth", 0.1)
                radii.append(min(w, d) / 2.0)  # approximate radius in XZ
                heights.append(h)
            else:
                radii.append(0.05)
                heights.append(0.05)

            # Velocity (vx, vy, vz)
            velocities.append((e.velocity[0], e.velocity[1], e.velocity[2]))
            smoke_amounts.append(e.smoke_amount)

        # Convert to wp arrays and return
        return EmitterData3D(
            centers=wp.array(centers, dtype=wp.vec3, device=self.device),
            radii=wp.array(radii, dtype=float, device=self.device),
            heights=wp.array(heights, dtype=float, device=self.device),
            velocities=wp.array(velocities, dtype=wp.vec3, device=self.device),
            smoke_amounts=wp.array(smoke_amounts, dtype=float, device=self.device),
            count=n
        )

    def _prepare_masks(self, masks: List["Mask"]) -> MaskData3D:
        """Convert Mask list to GPU arrays."""
        n = len(masks)
        if n == 0:
            return MaskData3D()

        centers = []
        radii = []

        for m in masks:
            centers.append((m.center[0], m.center[1], m.center[2]))

            if m.shape == "sphere":
                radii.append(m.params.get("radius", 0.05))
            elif m.shape == "cylinder":
                radii.append(m.params.get("radius", 0.05))
            elif m.shape == "box":
                w = m.params.get("width", 0.1)
                h = m.params.get("height", 0.1)
                d = m.params.get("depth", 0.1)
                radii.append(min(w, h, d) / 2.0)
            else:
                radii.append(0.05)

        # Convert to wp arrays and return
        return MaskData3D(
            centers=wp.array(centers, dtype=wp.vec3, device=self.device),
            radii=wp.array(radii, dtype=float, device=self.device),
            count=n
        )

    def _prepare_initial_velocities(self, initial_velocities: List["InitialVelocity"]) -> InitialVelocityData3D:
        """Convert InitialVelocity list to GPU arrays."""
        n = len(initial_velocities)
        if n == 0:
            return InitialVelocityData3D()

        centers = []
        half_sizes = []
        velocities = []

        for iv in initial_velocities:
            # Center position (x, y, z)
            centers.append((iv.center[0], iv.center[1], iv.center[2]))

            # Half sizes for box
            if iv.shape == "box":
                w = iv.params.get("width", 1.0)
                h = iv.params.get("height", 1.0)
                d = iv.params.get("depth", 1.0)
                half_sizes.append((w / 2.0, h / 2.0, d / 2.0))
            elif iv.shape == "sphere":
                r = iv.params.get("radius", 0.5)
                half_sizes.append((r, r, r))
            elif iv.shape == "cylinder":
                r = iv.params.get("radius", 0.5)
                h = iv.params.get("height", 1.0)
                half_sizes.append((r, h / 2.0, r))
            else:
                half_sizes.append((0.5, 0.5, 0.5))

            # Velocity (vx, vy, vz)
            velocities.append((iv.velocity[0], iv.velocity[1], iv.velocity[2]))

        return InitialVelocityData3D(
            centers=wp.array(centers, dtype=wp.vec3, device=self.device),
            half_sizes=wp.array(half_sizes, dtype=wp.vec3, device=self.device),
            velocities=wp.array(velocities, dtype=wp.vec3, device=self.device),
            count=n
        )

    def _prepare_bc(self, bc: "BoundaryCondition") -> dict:
        """Convert boundary condition types to integer flags."""
        bc_map = {"neumann": 0, "dirichlet": 1, "open": 2, "periodic": 3}
        return {
            "left": bc_map.get(bc.left, 0),
            "right": bc_map.get(bc.right, 0),
            "top": bc_map.get(bc.top, 0),
            "bottom": bc_map.get(bc.bottom, 0),
            "front": bc_map.get(bc.front, 0),
            "back": bc_map.get(bc.back, 0),
        }

    def step(self):
        """Advance the simulation by one time step."""
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
        """Reset the simulation grid to initial conditions."""
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

        # Apply initial velocity field (once at simulation start)
        if self.initial_velocity_data and self.initial_velocity_data.count > 0:
            wp.launch(
                kernel=apply_initial_velocity_kernel,
                dim=(self.nx + 1, self.ny + 1, self.nz + 1),
                inputs=[
                    self.grid.u0,
                    self.grid.v0,
                    self.grid.w0,
                    self.initial_velocity_data.centers,
                    self.initial_velocity_data.half_sizes,
                    self.initial_velocity_data.velocities,
                    self.initial_velocity_data.count,
                    self.nx,
                    self.ny,
                    self.nz,
                    self.grid.dx
                ]
            )

    def export_smoke_to_numpy(self, filename: str):
        """Export smoke density field to numpy .npy file."""
        smoke_np = self.grid.smoke0.numpy()
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, smoke_np)
