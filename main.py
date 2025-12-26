import argparse
import warp as wp
from core.simulation import SimulationController3D
from solvers.stable_fluid import StableFluidSolver3D
from core.slice_visualizer import run_visualization

wp.init()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Smoke Simulation")

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

    # Initialize Simulation
    sim = SimulationController3D(
        solver_type=StableFluidSolver3D,
        domain_size=(1.0, 1.0, 1.0),
        dx=args.dx,
        dt=args.dt,
        cfl_check=args.cfl_check,
        export=args.export,
        p_iter=args.p_iter,
    )

    # Run Visualization
    run_visualization(sim)