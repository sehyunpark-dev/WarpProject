"""
Volume Visualizer for Smoke Simulation using PyVista.

Loads numpy smoke density files and renders them as volume visualizations.
Supports single frame viewing and animation playback.

Usage:
    # Single frame:
    python -m core.volume_visualizer --folder outputs/numpy/251224_194029 --frame 50

    # Animation (all frames):
    python -m core.volume_visualizer --folder outputs/numpy/251224_194029 --animate

    # Animation with custom frame range and interval:
    python -m core.volume_visualizer --folder outputs/numpy/251224_194029 --animate --start 0 --end 100 --interval 50
"""

import argparse
import glob
import os
import numpy as np
import pyvista as pv


def load_smoke_frame(filepath: str) -> np.ndarray:
    """Load a single smoke density numpy file."""
    return np.load(filepath)


def create_volume_grid(smoke_data: np.ndarray) -> pv.ImageData:
    """
    Create a PyVista ImageData grid from smoke density data.

    Args:
        smoke_data: 3D numpy array of smoke density values

    Returns:
        PyVista ImageData with smoke density as point data
    """
    nx, ny, nz = smoke_data.shape

    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.spacing = (1.0 / nx, 1.0 / ny, 1.0 / nz)

    grid.cell_data["smoke"] = smoke_data.flatten(order="F")

    return grid


def create_smoke_opacity_transfer_function():
    """
    Create opacity transfer function for smoke visualization.
    0.0 (no smoke) -> fully transparent
    1.0 (dense smoke) -> opaque black
    """
    return [0.0, 0.0,   # density 0.0 -> opacity 0.0
            0.1, 0.1,   # density 0.1 -> opacity 0.1
            0.3, 0.3,   # density 0.3 -> opacity 0.3
            0.5, 0.5,   # density 0.5 -> opacity 0.5
            0.7, 0.7,   # density 0.7 -> opacity 0.7
            1.0, 1.0]   # density 1.0 -> opacity 1.0


def setup_camera_and_axes(plotter: pv.Plotter, grid: pv.ImageData):
    """
    Setup camera position and add bounding box with axes.
    Camera is positioned so that XZ plane is at the bottom and Y points up.

    Args:
        plotter: PyVista plotter instance
        grid: Volume grid to get bounds from
    """
    # Add bounding box with axes labels
    bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

    # Create box outline
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, style="wireframe", color="gray", line_width=1)

    # Add axes with labels at corners
    # X axis (red)
    plotter.add_mesh(
        pv.Line((bounds[0], bounds[2], bounds[4]), (bounds[1], bounds[2], bounds[4])),
        color="red", line_width=2
    )
    # Y axis (green) - pointing up
    plotter.add_mesh(
        pv.Line((bounds[0], bounds[2], bounds[4]), (bounds[0], bounds[3], bounds[4])),
        color="green", line_width=2
    )
    # Z axis (blue)
    plotter.add_mesh(
        pv.Line((bounds[0], bounds[2], bounds[4]), (bounds[0], bounds[2], bounds[5])),
        color="blue", line_width=2
    )

    # Add axis labels
    label_offset = 0.05
    plotter.add_point_labels(
        [(bounds[1] + label_offset, bounds[2], bounds[4])],
        ["X"], font_size=20, text_color="red", shape=None, always_visible=True
    )
    plotter.add_point_labels(
        [(bounds[0], bounds[3] + label_offset, bounds[4])],
        ["Y"], font_size=20, text_color="green", shape=None, always_visible=True
    )
    plotter.add_point_labels(
        [(bounds[0], bounds[2], bounds[5] + label_offset)],
        ["Z"], font_size=20, text_color="blue", shape=None, always_visible=True
    )

    # Set camera position: view from front-right-top, looking at center
    # XZ plane at bottom, Y pointing up
    center = grid.center
    # Camera position: offset in X, Y (up), and Z directions
    cam_distance = 2.5
    plotter.camera_position = [
        (center[0] + cam_distance * 0.8, center[1] + cam_distance * 0.5, center[2] + cam_distance),  # camera location
        center,  # focal point
        (0, 1, 0)  # view up vector (Y is up)
    ]


def visualize_single_frame(folder: str, frame: int):
    """
    Visualize a single frame of smoke simulation.

    Args:
        folder: Path to the folder containing numpy files
        frame: Frame number to visualize
    """
    filepath = os.path.join(folder, f"smoke_frame_{frame:05d}.npy")

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    print(f"Loading frame {frame} from {filepath}")
    smoke_data = load_smoke_frame(filepath)
    print(f"Smoke data shape: {smoke_data.shape}")
    print(f"Smoke density range: [{smoke_data.min():.4f}, {smoke_data.max():.4f}]")

    grid = create_volume_grid(smoke_data)

    plotter = pv.Plotter()
    plotter.set_background("white")

    # Volume rendering with black smoke on white background
    # Colormap: 0.0 -> white (invisible), 1.0 -> black (dense smoke)
    plotter.add_volume(
        grid,
        scalars="smoke",
        cmap="gray_r",  # Reversed gray: 0->white, 1->black
        opacity=create_smoke_opacity_transfer_function(),
        opacity_unit_distance=0.1,
        shade=False,
        clim=[0.0, 1.0],
    )

    # Setup camera and axes (XZ plane at bottom, Y up)
    setup_camera_and_axes(plotter, grid)

    plotter.add_text(f"Frame: {frame}", position="upper_left", font_size=12, color="black")
    plotter.show()


def visualize_animation(folder: str, start: int = None, end: int = None, interval: int = 100):
    """
    Visualize animation of smoke simulation.

    Args:
        folder: Path to the folder containing numpy files
        start: Starting frame number (default: first available)
        end: Ending frame number (default: last available)
        interval: Time interval between frames in milliseconds
    """
    # Find all available frames
    pattern = os.path.join(folder, "smoke_frame_*.npy")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Error: No numpy files found in {folder}")
        return

    # Extract frame numbers
    frame_numbers = []
    for f in files:
        basename = os.path.basename(f)
        frame_num = int(basename.replace("smoke_frame_", "").replace(".npy", ""))
        frame_numbers.append(frame_num)

    frame_numbers = sorted(frame_numbers)

    # Apply range filter
    if start is not None:
        frame_numbers = [f for f in frame_numbers if f >= start]
    if end is not None:
        frame_numbers = [f for f in frame_numbers if f <= end]

    if not frame_numbers:
        print(f"Error: No frames found in specified range")
        return

    print(f"Found {len(frame_numbers)} frames: {frame_numbers[0]} to {frame_numbers[-1]}")

    # Load first frame to get dimensions
    first_file = os.path.join(folder, f"smoke_frame_{frame_numbers[0]:05d}.npy")
    first_data = load_smoke_frame(first_file)
    print(f"Smoke data shape: {first_data.shape}")

    # Preload all frames for smoother animation
    print("Preloading frames...")
    all_grids = []
    for frame_num in frame_numbers:
        filepath = os.path.join(folder, f"smoke_frame_{frame_num:05d}.npy")
        smoke_data = load_smoke_frame(filepath)
        grid = create_volume_grid(smoke_data)
        all_grids.append(grid)
    print("Preloading complete.")

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Animation state
    current_idx = [0]
    is_playing = [True]
    volume_actor = [None]

    def add_volume_for_frame(idx):
        """Add volume actor for given frame index."""
        if volume_actor[0] is not None:
            plotter.remove_actor(volume_actor[0])

        volume_actor[0] = plotter.add_volume(
            all_grids[idx],
            scalars="smoke",
            cmap="gray_r",
            opacity=create_smoke_opacity_transfer_function(),
            opacity_unit_distance=0.1,
            shade=False,
            clim=[0.0, 1.0],
        )

    # Add initial volume
    add_volume_for_frame(0)

    # Setup camera and axes (XZ plane at bottom, Y up)
    setup_camera_and_axes(plotter, all_grids[0])

    plotter.add_text(
        f"Frame: {frame_numbers[0]}",
        position="upper_left",
        font_size=12,
        color="black",
        name="frame_text"
    )

    def update_frame(caller, event):
        """Timer callback to update frame."""
        if not is_playing[0]:
            return

        current_idx[0] = (current_idx[0] + 1) % len(frame_numbers)
        frame_num = frame_numbers[current_idx[0]]

        add_volume_for_frame(current_idx[0])

        plotter.remove_actor("frame_text")
        plotter.add_text(
            f"Frame: {frame_num}",
            position="upper_left",
            font_size=12,
            color="black",
            name="frame_text"
        )

        plotter.render()

    def toggle_play():
        """Toggle play/pause."""
        is_playing[0] = not is_playing[0]
        status = "Playing" if is_playing[0] else "Paused"
        print(f"Animation: {status}")

    # Add keyboard callback for play/pause
    plotter.add_key_event("space", toggle_play)

    print("Controls:")
    print("  Space: Play/Pause animation")
    print("  Q: Quit")

    # Setup timer using VTK interactor
    plotter.show(interactive_update=True, auto_close=False)

    # Create timer observer
    interactor = plotter.iren.interactor
    interactor.AddObserver("TimerEvent", update_frame)
    timer_id = interactor.CreateRepeatingTimer(interval)

    # Start interaction loop
    interactor.Start()


def main():
    parser = argparse.ArgumentParser(
        description="Volume visualizer for smoke simulation using PyVista",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single frame:
    python -m core.volume_visualizer --folder outputs/numpy/251224_194029 --frame 50

  Animation (all frames):
    python -m core.volume_visualizer --folder outputs/numpy/251224_194029 --animate

  Animation with range:
    python -m core.volume_visualizer --folder outputs/numpy/251224_194029 --animate --start 0 --end 100
        """
    )

    parser.add_argument(
        "--folder", "-f",
        type=str,
        required=True,
        help="Path to folder containing numpy smoke files (e.g., outputs/numpy/251224_194029)"
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame number to visualize (for single frame mode)"
    )

    parser.add_argument(
        "--animate", "-a",
        action="store_true",
        help="Enable animation mode to play through frames"
    )

    parser.add_argument(
        "--start", "-s",
        type=int,
        default=None,
        help="Starting frame number for animation (default: first available)"
    )

    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="Ending frame number for animation (default: last available)"
    )

    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=100,
        help="Time interval between frames in milliseconds (default: 100)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.animate and args.frame is None:
        parser.error("Either --frame or --animate must be specified")

    if args.animate and args.frame is not None:
        print("Warning: --frame is ignored when --animate is specified")

    # Check if folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        return

    # Run visualization
    if args.animate:
        visualize_animation(args.folder, args.start, args.end, args.interval)
    else:
        visualize_single_frame(args.folder, args.frame)


if __name__ == "__main__":
    main()
