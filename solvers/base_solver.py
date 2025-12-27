from abc import ABC, abstractmethod
from typing import Union
from core.mac_grid_3d import MACGrid3D
from core.mac_grid_2d import MACGrid2D

# Type alias for grid types
GridType = Union[MACGrid3D, MACGrid2D]

class Solver(ABC):
    @abstractmethod
    def __init__(self, grid: GridType, dt: float, **kwargs):
        pass

    @abstractmethod
    def step(self):
        pass