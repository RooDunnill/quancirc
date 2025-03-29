from ..circuit_config import *
from .base_class_utilities.base_class_errors import BaseGateError

class BaseGate:

    def __getitem__(self, index: tuple[int, int]):
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.matrix[row, col]