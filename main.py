import argparse
import warp as wp
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
        "--dim",
        type=int,
        choices=[2, 3],
        default=3,
        help="Simulation dimension: 2 for 2D, 3 for 3D (default: 3)"
    )
    parser.add_argument(
        "--cfl-check",
        action="store_true",
        help="Enable CFL number checking"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Enable numpy export"
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1.0/256.0,
        help="Grid spacing (default: 1/256)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Time step (default: 0.005)"
    )
    parser.add_argument(
        "--p-iter",
        type=int,
        default=100,
        help="Pressure solver iterations (default: 100)"
    )

    args = parser.parse_args()

    if args.dim == 3:
        # 3D Simulation
        sim = SimulationController3D(
            solver_type=StableFluidSolver3D,
            domain_size=(1.0, 1.0, 1.0),
            dx=args.dx,
            dt=args.dt,
            cfl_check=args.cfl_check,
            export=args.export,
            p_iter=args.p_iter,
        )
        run_visualization_3d(sim)
    else:
        # 2D Simulation
        sim = SimulationController2D(
            solver_type=StableFluidSolver2D,
            domain_size=(1.0, 1.0),
            dx=args.dx,
            dt=args.dt,
            cfl_check=args.cfl_check,
            export=args.export,
            p_iter=args.p_iter,
        )
        run_visualization_2d(sim)
