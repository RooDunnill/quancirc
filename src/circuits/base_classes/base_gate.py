from ..circuit_config import *
from .base_class_utilities.base_class_errors import BaseGateError

class BaseGate:
    def __init__(self, **kwargs):
        self.skip_val = kwargs.get("skip_validation", False)
        self.name = kwargs.get("name", "QG")
        self.matrix = kwargs.get("matrix", None)

    def __getitem__(self, index: tuple[int, int]):
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.matrix[row, col]
        
    def __imod__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        if isinstance(other, BaseGate):
            if self.immutable:
                raise BaseGateError(f"This operation is not valid for an immutable object")
            self = self % other
            return self
        raise BaseGateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected types Qubit and Qubit")