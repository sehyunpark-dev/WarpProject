import warp as wp
from core.simulation import SimulationController3D
from solvers.stable_fluid import StableFluidSolver3D
from core.slice_visualizer import run_visualization

wp.init()

if __name__ == "__main__":
    # Initialize Simulation
    sim = SimulationController3D(
        solver_type=StableFluidSolver3D,
        domain_size=(1.0, 1.0, 1.0),
        dx=1.0/256.0,
        dt=0.01
    )
    
    # Run Visualization
    run_visualization(sim)