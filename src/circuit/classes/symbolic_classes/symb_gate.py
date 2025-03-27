import numpy as np
from ...circuit_utilities.circuit_errors import SymbGateError
from ...circuit_config import *
import sympy as sp


class SymbGate:
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'symbgate')
        self.matrix = kwargs.get("matrix", None)
        self.dim: int = self.matrix.shape[0]
        self.length = self.dim ** 2
        self.n: int =  int(np.log2(self.dim))

    def __str__(self) -> str:
        return f"{self.name}\n{self.matrix}"
    
    def __mod__(self, other) -> "SymbGate":            #tensor product
        if isinstance(other, SymbGate):
            new_matrix = sp.tensorproduct(self.matrix, other.matrix)
            kwargs = {"matrix": new_matrix}
            return SymbGate(**kwargs)
        else:
            raise SymbGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
        
    def __matmul__(self, other) -> "SymbGate":
        if isinstance(other, SymbGate):
            new_matrix = self.matrix @ other.matrix
            kwargs = {"matrix": new_matrix}
            return SymbGate(**kwargs)
        elif other.class_type == "symbqubit":
            new_rho = self.matrix * other.rho * sp.conjugate(self.matrix.T)
            kwargs = {"rho": new_rho}
            return other.__class__(**kwargs)
        raise SymbGateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected type SymbGate, SymbQubit")
