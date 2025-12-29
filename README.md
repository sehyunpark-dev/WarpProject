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
├── main.py                             # Entry point
├── scene_parser.py                     # JSON scene file parser
├── core/
│   ├── dim2d/                          # 2D simulation modules
│   │   ├── mac_grid_2d.py              # 2D MAC grid data structure and sampling functions
│   │   ├── simulation_2d.py            # 2D simulation controller
│   │   └── visualizer_2d.py            # 2D visualization (Matplotlib)
│   └── dim3d/                          # 3D simulation modules
│       ├── mac_grid_3d.py              # 3D MAC grid data structure and sampling functions
│       ├── simulation_3d.py            # 3D simulation controller
│       ├── slice_visualizer_3d.py      # 3D slice visualization (Matplotlib)
│       └── volume_visualizer.py        # 3D volume rendering (PyVista)
├── solvers/
│   ├── base_solver.py                  # Solver interface
│   ├── stable_fluid_2d.py              # 2D Stable Fluids implementation
│   └── stable_fluid_3d.py              # 3D Stable Fluids implementation
├── scenes/                             # Scene configuration files
│   ├── basic_2d.json                   # Basic 2D smoke simulation
│   ├── basic_3d.json                   # Basic 3D smoke simulation
│   └── karman_vortex.json              # Karman vortex street example
└── outputs/
    └── numpy/                          # Exported simulation frames
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
pip install -r requirements.txt
```

## Usage

### Running the Simulation

Simulations are configured using JSON scene files. Run a simulation by specifying a scene file:

```bash
# Run 2D simulation
python main.py --scene scenes/basic_2d.json

# Run 3D simulation
python main.py --scene scenes/basic_3d.json
```

### Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--scene` | Path to scene JSON file | Yes |

### Scene File Format

Scene files are JSON files that define all simulation parameters. Example structure:

```json
{
    "scene": {
        "dimension": 2,
        "domain_size": [1.0, 1.0],
        "dx": 0.005,
        "cfl_check": false,
        "export": false,
        "bc": {
            "left": "neumann",
            "right": "neumann",
            "top": "neumann",
            "bottom": "neumann"
        }
    },
    "solver": {
        "type": "stable_fluid",
        "dt": 0.005,
        "p_iter": 100,
        "rho": 1.0,
        "nu": 0.0
    },
    "emitters": [
        {
            "name": "bottom_smoke",
            "shape": "circle",
            "center": [0.5, 0.05],
            "params": [0.05],
            "velocity": [0.0, 1.0],
            "smoke_amount": 1.0
        }
    ],
    "masks": []
}
```

### Scene Configuration Reference

#### Scene Settings

| Field | Description | Default |
|-------|-------------|---------|
| `dimension` | Simulation dimension (2 or 3) | 3 |
| `domain_size` | Physical domain size in meters | [1.0, 1.0, 1.0] |
| `dx` | Grid cell spacing | 0.01 |
| `cfl_check` | Enable CFL number checking | false |
| `export` | Enable numpy frame export | false |
| `bc` | Boundary conditions (neumann, dirichlet, open, periodic) | neumann |

#### Solver Settings

| Field | Description | Default |
|-------|-------------|---------|
| `type` | Solver type | "stable_fluid" |
| `dt` | Time step size (seconds) | 0.01 |
| `p_iter` | Pressure solver iterations | 100 |
| `rho` | Fluid density (kg/m³) | 1.0 |
| `nu` | Kinematic viscosity (m²/s) | 0.0 |

#### Emitter Settings

| Field | Description |
|-------|-------------|
| `name` | Unique identifier |
| `shape` | Shape type: circle, rectangle (2D) / sphere, box, cylinder (3D) |
| `center` | Position in world coordinates |
| `params` | Shape parameters (e.g., [radius] for circle, [radius, height] for cylinder) |
| `velocity` | Velocity to inject |
| `smoke_amount` | Smoke density (0.0 to 1.0) |

#### Mask Settings (Obstacles)

| Field | Description |
|-------|-------------|
| `name` | Unique identifier |
| `shape` | Shape type (same as emitters) |
| `center` | Position in world coordinates |
| `params` | Shape parameters |

### Testing Scene Parser

You can test the scene parser directly:

```bash
python scene_parser.py scenes/basic_2d.json
```

### Volume Visualization (3D only)

After exporting frames (set `"export": true` in scene file), use the volume visualizer:

```bash
# Single frame
python -m core.dim3d.volume_visualizer --folder outputs/numpy/YYMMDD_HHMMSS --frame 50

# Animation (all frames)
python -m core.dim3d.volume_visualizer --folder outputs/numpy/YYMMDD_HHMMSS --animate

# Animation with frame range
python -m core.dim3d.volume_visualizer --folder outputs/numpy/YYMMDD_HHMMSS --animate --start 0 --end 100 --interval 50
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
