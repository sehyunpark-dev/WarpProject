# WarpProject

GPU-accelerated 2D/3D smoke simulation using [NVIDIA Warp](https://github.com/NVIDIA/warp).

## Overview

This project implements real-time fluid simulations based on the Stable Fluids algorithm (Stam, 1999). The simulation runs entirely on the GPU using NVIDIA Warp, enabling high-resolution smoke simulations with interactive performance.

### Features

- **2D & 3D Stable Fluid Solver**: Semi-Lagrangian advection with pressure projection
- **MAC Grid**: Staggered grid (Marker-and-Cell) for accurate velocity representation
- **GPU Acceleration**: All computations run on CUDA via NVIDIA Warp
- **Real-time Visualization**: 2D viewer and 3D slice viewer using Matplotlib
- **Volume Rendering**: 3D visualization using PyVista
- **Numpy Export**: Save simulation frames for post-processing

## Project Structure

```
WarpProject/
├── main.py                         # Entry point
├── core/
│   ├── mac_grid_2d.py              # 2D MAC grid data structure and sampling functions
│   ├── mac_grid_3d.py              # 3D MAC grid data structure and sampling functions
│   ├── simulation_2d.py            # 2D simulation controller
│   ├── simulation_3d.py            # 3D simulation controller
│   ├── visualizer_2d.py            # 2D visualization (Matplotlib)
│   ├── slice_visualizer_3d.py      # 3D slice visualization (Matplotlib)
│   └── volume_visualizer.py        # 3D volume rendering (PyVista)
├── solvers/
    ├── base_solver.py              # Solver interface
    ├── stable_fluid_2d.py          # 2D Stable Fluids implementation
    └── stable_fluid_3d.py          # 3D Stable Fluids implementation
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

Basic simulation with real-time visualization:

```bash
# 3D simulation (default)
python main.py

# 2D simulation
python main.py --dim 2
```

With optional flags:

```bash
# Enable CFL number checking
python main.py --cfl-check

# Enable numpy export for later visualization
python main.py --export

# 2D simulation with CFL check and export
python main.py --dim 2 --cfl-check --export

# Custom simulation parameters
python main.py --dx 0.00390625 --dt 0.01 --p-iter 50
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dim` | Simulation dimension (2 or 3) | 3 |
| `--cfl-check` | Enable CFL number checking | Disabled |
| `--export` | Enable numpy frame export | Disabled |
| `--dx` | Grid spacing | 1/256 |
| `--dt` | Time step | 0.005 |
| `--p-iter` | Pressure solver iterations | 100 |

### Volume Visualization (3D only)

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
