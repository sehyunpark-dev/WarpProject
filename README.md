# WarpProject

GPU-accelerated 3D smoke simulation using [NVIDIA Warp](https://github.com/NVIDIA/warp).

## Overview

This project implements a real-time 3D fluid simulation based on the Stable Fluids algorithm (Stam, 1999). The simulation runs entirely on the GPU using NVIDIA Warp, enabling high-resolution smoke simulations with interactive performance.

### Features

- **3D Stable Fluid Solver**: Semi-Lagrangian advection with pressure projection
- **MAC Grid**: Staggered grid (Marker-and-Cell) for accurate velocity representation
- **GPU Acceleration**: All computations run on CUDA via NVIDIA Warp
- **Real-time Visualization**: 2D slice viewer using Matplotlib
- **Volume Rendering**: 3D visualization using PyVista
- **Numpy Export**: Save simulation frames for post-processing

## Project Structure

```
WarpProject/
├── main.py                     # Entry point
├── core/
│   ├── grid.py                 # MAC grid data structure and sampling functions
│   ├── simulation.py           # Simulation controller
│   ├── slice_visualizer.py     # 2D slice visualization (Matplotlib)
│   └── volume_visualizer.py    # 3D volume rendering (PyVista)
├── solvers/
│   ├── base_solver.py          # Solver interface
│   └── stable_fluid.py         # Stable Fluids implementation
└── outputs/
    └── numpy/                  # Exported simulation frames
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- NVIDIA Warp
- NumPy
- Matplotlib
- PyVista (for volume rendering)

### Installation

```bash
pip install warp-lang numpy matplotlib pyvista
```

## Usage

### Running the Simulation

Basic simulation with real-time 2D slice visualization:

```bash
python main.py
```

With optional flags:

```bash
# Enable CFL number checking
python main.py --cfl-check

# Enable numpy export for later visualization
python main.py --export

# Both options
python main.py --cfl-check --export

# Custom simulation parameters
python main.py --dx 0.00390625 --dt 0.01 --p-iter 50
```

#### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--cfl-check` | Enable CFL number checking | Disabled |
| `--export` | Enable numpy frame export | Disabled |
| `--dx` | Grid spacing | 1/256 |
| `--dt` | Time step | 0.005 |
| `--p-iter` | Pressure solver iterations | 100 |

### Volume Visualization

After exporting frames with `--export`, use the volume visualizer:

```bash
# Single frame
python -m core.volume_visualizer --folder outputs/numpy/YYMMDD_HHMMSS --frame 50

# Animation (all frames)
python -m core.volume_visualizer --folder outputs/numpy/YYMMDD_HHMMSS --animate

# Animation with frame range
python -m core.volume_visualizer --folder outputs/numpy/YYMMDD_HHMMSS --animate --start 0 --end 100 --interval 50
```

#### Volume Visualizer Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--folder`, `-f` | Path to numpy export folder | Required |
| `--frame` | Single frame number to view | - |
| `--animate`, `-a` | Enable animation mode | Disabled |
| `--start`, `-s` | Start frame for animation | First available |
| `--end`, `-e` | End frame for animation | Last available |
| `--interval`, `-i` | Frame interval (ms) | 100 |

#### Controls (Animation Mode)

- **Space**: Play/Pause
- **Q**: Quit

## Algorithm

### Stable Fluids Pipeline

Each simulation step follows the classic Stable Fluids algorithm:

1. **External Forces**: Inject smoke and velocity at source region
2. **Advection**: Semi-Lagrangian advection for velocity and density
3. **Projection**: Pressure solve (Jacobi iteration) to enforce incompressibility

### Grid Structure

The simulation uses a MAC (Marker-and-Cell) staggered grid:

- **Cell Centers**: Pressure, smoke density, divergence
- **Face Centers**: Velocity components (u, v, w)

### Boundary Conditions

- **Velocity**: No-penetration (normal = 0), free-slip (tangential unchanged)
- **Pressure**: Neumann boundary conditions

## Simulation Parameters

### Source Configuration

The smoke source is a cylinder at the bottom of the domain:

- **Position**: Center at (0.5, 0.05, 0.5) in world units
- **Radius**: 0.05 (XZ plane)
- **Height**: 0.05 (Y direction)
- **Velocity**: 100 m/s upward injection

### Domain

- **Size**: 1.0 x 1.0 x 1.0 (unit cube)
- **Default Resolution**: 256 x 256 x 256 cells

## License

MIT License
