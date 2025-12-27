"""
Scene Parser for WarpProject

This module provides a parser for JSON scene files that define simulation configurations.
It handles 2D and 3D simulations with support for various emitter shapes and obstacle masks.

Usage:
    from scene_parser import SceneParser

    parser = SceneParser("scenes/basic_2d.json")
    config = parser.parse()
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path


# =============================================================================
# Shape Parameter Schema
# =============================================================================

# Defines the expected parameter names for each shape type.
# This allows consistent parsing regardless of 2D or 3D dimension.
SHAPE_SCHEMA: Dict[str, List[str]] = {
    # 2D shapes
    "circle": ["radius"],
    "rectangle": ["width", "height"],

    # 3D shapes
    "sphere": ["radius"],
    "box": ["width", "height", "depth"],
    "cylinder": ["radius", "height"],
}


# =============================================================================
# Data Classes for Parsed Configuration
# =============================================================================

@dataclass
class BoundaryCondition:
    """
    Represents boundary conditions for the simulation domain.

    For 2D: left, right, top, bottom
    For 3D: left, right, top, bottom, front, back

    Supported BC types:
        - "neumann": Zero gradient (free-slip wall)
        - "dirichlet": Fixed value (no-slip wall)
        - "open": Open boundary (inlet/outlet)
        - "periodic": Periodic boundary
    """
    left: str = "neumann"
    right: str = "neumann"
    top: str = "neumann"
    bottom: str = "neumann"
    front: str = "neumann"  # 3D only
    back: str = "neumann"   # 3D only


@dataclass
class SceneConfig:
    """
    Configuration for the simulation scene/domain.

    Attributes:
        dimension: 2 or 3 for 2D/3D simulation
        domain_size: Physical size of the domain (meters)
        dx: Grid cell size (meters)
        cfl_check: Whether to compute and display CFL number
        export: Whether to export simulation data to files
        bc: Boundary conditions for each domain face
    """
    dimension: int
    domain_size: Tuple[float, ...]
    dx: float
    cfl_check: bool = False
    export: bool = False
    bc: BoundaryCondition = field(default_factory=BoundaryCondition)


@dataclass
class SolverConfig:
    """
    Configuration for the fluid solver.

    Attributes:
        type: Solver type (e.g., "stable_fluid")
        dt: Time step size (seconds)
        p_iter: Number of pressure solver iterations
        rho: Fluid density (kg/m^3)
        nu: Kinematic viscosity (m^2/s), 0 for inviscid flow
    """
    type: str
    dt: float
    p_iter: int = 100
    rho: float = 1.0
    nu: float = 0.0


@dataclass
class Emitter:
    """
    Configuration for a smoke/velocity emitter.

    Emitters inject smoke density and/or velocity into the simulation.

    Attributes:
        name: Unique identifier for the emitter
        shape: Shape type ("circle", "rectangle", "sphere", "box", "cylinder")
        center: Position of the emitter center in world coordinates
        params: Shape-specific parameters (see SHAPE_SCHEMA)
        velocity: Velocity to inject [vx, vy] or [vx, vy, vz]
        smoke_amount: Smoke density to inject (0.0 to 1.0)
    """
    name: str
    shape: str
    center: Tuple[float, ...]
    params: Dict[str, float]
    velocity: Tuple[float, ...]
    smoke_amount: float = 1.0


@dataclass
class Mask:
    """
    Configuration for an obstacle mask (solid region).

    Masks define solid regions where fluid cannot flow.
    Velocity is set to zero inside masks (no-slip condition).

    Attributes:
        name: Unique identifier for the mask
        shape: Shape type ("circle", "rectangle", "sphere", "box", "cylinder")
        center: Position of the mask center in world coordinates
        params: Shape-specific parameters (see SHAPE_SCHEMA)
    """
    name: str
    shape: str
    center: Tuple[float, ...]
    params: Dict[str, float]


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration parsed from a JSON scene file.

    This is the main output of the SceneParser, containing all
    settings needed to initialize and run a simulation.

    Attributes:
        scene: Domain and environment settings
        solver: Solver parameters
        emitters: List of smoke/velocity sources
        masks: List of solid obstacles
    """
    scene: SceneConfig
    solver: SolverConfig
    emitters: List[Emitter] = field(default_factory=list)
    masks: List[Mask] = field(default_factory=list)


# =============================================================================
# Scene Parser Class
# =============================================================================

class SceneParser:
    """
    Parser for JSON scene configuration files.

    Reads a JSON file and converts it into structured dataclasses
    that can be used to initialize a simulation.

    Example:
        >>> parser = SceneParser("scenes/karman_vortex.json")
        >>> config = parser.parse()
        >>> print(config.scene.dimension)  # 2
        >>> print(config.emitters[0].name) # "inlet"
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize the parser with a path to a JSON scene file.

        Args:
            filepath: Path to the JSON scene file

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"Scene file not found: {self.filepath}")

        self._raw_data: Dict[str, Any] = {}

    def _load_json(self) -> Dict[str, Any]:
        """
        Load and parse the JSON file.

        Returns:
            Dictionary containing the raw JSON data

        Raises:
            json.JSONDecodeError: If the file contains invalid JSON
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _parse_params(self, shape: str, params: List[float]) -> Dict[str, float]:
        """
        Convert a params array to a named dictionary based on shape type.

        Args:
            shape: Shape type (e.g., "circle", "cylinder")
            params: List of numeric parameters

        Returns:
            Dictionary mapping parameter names to values

        Raises:
            ValueError: If shape is unknown or params count doesn't match

        Example:
            >>> self._parse_params("cylinder", [0.05, 0.1])
            {"radius": 0.05, "height": 0.1}
        """
        if shape not in SHAPE_SCHEMA:
            raise ValueError(f"Unknown shape type: {shape}. "
                           f"Supported shapes: {list(SHAPE_SCHEMA.keys())}")

        expected_keys = SHAPE_SCHEMA[shape]

        if len(params) != len(expected_keys):
            raise ValueError(
                f"Shape '{shape}' requires {len(expected_keys)} params "
                f"({expected_keys}), but got {len(params)}: {params}"
            )

        return dict(zip(expected_keys, params))

    def _parse_boundary_conditions(self, bc_data: Dict[str, str], dimension: int) -> BoundaryCondition:
        """
        Parse boundary condition settings.

        Args:
            bc_data: Dictionary of boundary conditions from JSON
            dimension: Simulation dimension (2 or 3)

        Returns:
            BoundaryCondition dataclass instance
        """
        bc = BoundaryCondition()

        # Common boundaries for both 2D and 3D
        bc.left = bc_data.get("left", "neumann")
        bc.right = bc_data.get("right", "neumann")
        bc.top = bc_data.get("top", "neumann")
        bc.bottom = bc_data.get("bottom", "neumann")

        # Additional boundaries for 3D
        if dimension == 3:
            bc.front = bc_data.get("front", "neumann")
            bc.back = bc_data.get("back", "neumann")

        return bc

    def _parse_scene(self, scene_data: Dict[str, Any]) -> SceneConfig:
        """
        Parse the scene configuration section.

        Args:
            scene_data: Dictionary containing scene settings

        Returns:
            SceneConfig dataclass instance
        """
        dimension = scene_data.get("dimension", 3)

        # Parse boundary conditions
        bc_data = scene_data.get("bc", {})
        bc = self._parse_boundary_conditions(bc_data, dimension)

        return SceneConfig(
            dimension   = dimension,
            domain_size = tuple(scene_data.get("domain_size", [1.0, 1.0, 1.0])),
            dx          = scene_data.get("dx", 0.01),
            cfl_check   = scene_data.get("cfl_check", False),
            export      = scene_data.get("export", False),
            bc          = bc
        )

    def _parse_solver(self, solver_data: Dict[str, Any]) -> SolverConfig:
        """
        Parse the solver configuration section.

        Args:
            solver_data: Dictionary containing solver settings

        Returns:
            SolverConfig dataclass instance
        """
        return SolverConfig(
            type   = solver_data.get("type", "stable_fluid"),
            dt     = solver_data.get("dt", 0.01),
            p_iter = solver_data.get("p_iter", 100),
            rho    = solver_data.get("rho", 1.0),
            nu     = solver_data.get("nu", 0.0)
        )

    def _parse_emitter(self, emitter_data: Dict[str, Any]) -> Emitter:
        """
        Parse a single emitter configuration.

        Args:
            emitter_data: Dictionary containing emitter settings

        Returns:
            Emitter dataclass instance
        """
        shape = emitter_data.get("shape", "circle")
        raw_params = emitter_data.get("params", [])

        return Emitter(
            name         = emitter_data.get("name", "unnamed_emitter"),
            shape        = shape,
            center       = tuple(emitter_data.get("center", [0.0, 0.0])),
            params       = self._parse_params(shape, raw_params),
            velocity     = tuple(emitter_data.get("velocity", [0.0, 0.0])),
            smoke_amount = emitter_data.get("smoke_amount", 1.0)
        )

    def _parse_mask(self, mask_data: Dict[str, Any]) -> Mask:
        """
        Parse a single mask (obstacle) configuration.

        Args:
            mask_data: Dictionary containing mask settings

        Returns:
            Mask dataclass instance
        """
        shape = mask_data.get("shape", "circle")
        raw_params = mask_data.get("params", [])

        return Mask(
            name   = mask_data.get("name", "unnamed_mask"),
            shape  = shape,
            center = tuple(mask_data.get("center", [0.0, 0.0])),
            params = self._parse_params(shape, raw_params)
        )

    def parse(self) -> SimulationConfig:
        """
        Parse the JSON scene file and return a complete simulation configuration.

        This is the main entry point for parsing a scene file.

        Returns:
            SimulationConfig containing all parsed settings

        Raises:
            FileNotFoundError: If scene file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            ValueError: If required fields are missing or invalid

        Example:
            >>> parser = SceneParser("scenes/basic_2d.json")
            >>> config = parser.parse()
            >>> print(f"Domain: {config.scene.domain_size}")
            >>> print(f"Solver: {config.solver.type}")
        """
        # Load raw JSON data
        self._raw_data = self._load_json()

        # Parse each section
        scene = self._parse_scene(self._raw_data.get("scene", {}))
        solver = self._parse_solver(self._raw_data.get("solver", {}))

        # Parse emitters list
        emitters = [
            self._parse_emitter(e)
            for e in self._raw_data.get("emitters", [])
        ]

        # Parse masks list
        masks = [
            self._parse_mask(m)
            for m in self._raw_data.get("masks", [])
        ]

        return SimulationConfig(
            scene=scene,
            solver=solver,
            emitters=emitters,
            masks=masks
        )

    def get_raw_data(self) -> Dict[str, Any]:
        """
        Get the raw JSON data after parsing.

        Useful for debugging or accessing fields not covered by dataclasses.

        Returns:
            Dictionary containing the original JSON data
        """
        return self._raw_data


# =============================================================================
# Utility Functions
# =============================================================================

def load_scene(filepath: Union[str, Path]) -> SimulationConfig:
    """
    Convenience function to load and parse a scene file in one call.

    Args:
        filepath: Path to the JSON scene file

    Returns:
        Parsed SimulationConfig

    Example:
        >>> config = load_scene("scenes/karman_vortex.json")
    """
    parser = SceneParser(filepath)
    return parser.parse()


def get_grid_size(config: SimulationConfig) -> Tuple[int, ...]:
    """
    Calculate the grid dimensions from scene configuration.

    Args:
        config: Parsed simulation configuration

    Returns:
        Tuple of grid dimensions (nx, ny) or (nx, ny, nz)

    Example:
        >>> config = load_scene("scenes/basic_2d.json")
        >>> nx, ny = get_grid_size(config)
        >>> print(f"Grid: {nx} x {ny}")
    """
    dx = config.scene.dx
    domain_size = config.scene.domain_size

    return tuple(int(size / dx) for size in domain_size)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    # Default test file
    test_file = "scenes/basic_2d.json"

    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    print(f"Parsing: {test_file}")
    print("=" * 60)

    try:
        config = load_scene(test_file)

        # Print scene info
        print(f"\n[Scene]")
        print(f"  Dimension: {config.scene.dimension}D")
        print(f"  Domain Size: {config.scene.domain_size}")
        print(f"  Grid Resolution (dx): {config.scene.dx}")
        print(f"  Grid Size: {get_grid_size(config)}")
        print(f"  CFL Check: {config.scene.cfl_check}")
        print(f"  Export: {config.scene.export}")

        # Print boundary conditions
        print(f"\n[Boundary Conditions]")
        bc = config.scene.bc
        print(f"  Left: {bc.left}, Right: {bc.right}")
        print(f"  Top: {bc.top}, Bottom: {bc.bottom}")
        if config.scene.dimension == 3:
            print(f"  Front: {bc.front}, Back: {bc.back}")

        # Print solver info
        print(f"\n[Solver]")
        print(f"  Type: {config.solver.type}")
        print(f"  Time Step (dt): {config.solver.dt}")
        print(f"  Pressure Iterations: {config.solver.p_iter}")
        print(f"  Density (rho): {config.solver.rho}")
        print(f"  Viscosity (nu): {config.solver.nu}")

        # Print emitters
        print(f"\n[Emitters] ({len(config.emitters)} total)")
        for i, emitter in enumerate(config.emitters):
            print(f"  [{i}] {emitter.name}")
            print(f"      Shape: {emitter.shape}")
            print(f"      Center: {emitter.center}")
            print(f"      Params: {emitter.params}")
            print(f"      Velocity: {emitter.velocity}")
            print(f"      Smoke Amount: {emitter.smoke_amount}")

        # Print masks
        print(f"\n[Masks] ({len(config.masks)} total)")
        for i, mask in enumerate(config.masks):
            print(f"  [{i}] {mask.name}")
            print(f"      Shape: {mask.shape}")
            print(f"      Center: {mask.center}")
            print(f"      Params: {mask.params}")

        print("\n" + "=" * 60)
        print("Parsing completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
