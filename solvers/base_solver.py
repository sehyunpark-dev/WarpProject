from abc import ABC, abstractmethod
from core.grid import MACGrid3D

class Solver(ABC):
    @abstractmethod
    def __init__(self, grid: MACGrid3D, dt: float, **kwargs):
        pass

    @abstractmethod
    def step(self):
        pass