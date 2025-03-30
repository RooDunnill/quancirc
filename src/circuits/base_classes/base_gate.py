from ..circuit_config import *
from .base_class_utilities.base_class_errors import BaseGateError

class BaseGate:
    def __init__(self, **kwargs):
        self.skip_val = kwargs.get("skip_validation", False)
        self.name = kwargs.get("name", "Quantum Gate")
        self.matrix = kwargs.get("matrix", None)

    def __getitem__(self, index: tuple[int, int]):
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.matrix[row, col]