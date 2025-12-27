import argparse
import warp as wp

from scene_parser import load_scene
from core.dim3d.simulation_3d import SimulationController3D
from core.dim2d.simulation_2d import SimulationController2D
from solvers.stable_fluid_3d import StableFluidSolver3D
from solvers.stable_fluid_2d import StableFluidSolver2D
from core.dim3d.slice_visualizer_3d import run_visualization as run_visualization_3d
from core.dim2d.visualizer_2d import run_visualization as run_visualization_2d

wp.init()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke Simulation")

    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to scene JSON file (e.g., scenes/basic_2d.json)"
    )

    args = parser.parse_args()

    # Load and parse scene configuration
    print(f"Loading scene: {args.scene}")
    config = load_scene(args.scene)

    # Select solver based on dimension
    if config.scene.dimension == 3:
        solver_type = StableFluidSolver3D
        sim = SimulationController3D(
            solver_type=solver_type,
            config=config
        )
        run_visualization_3d(sim)
    else:
        solver_type = StableFluidSolver2D
        sim = SimulationController2D(
            solver_type=solver_type,
            config=config
        )
        run_visualization_2d(sim)
