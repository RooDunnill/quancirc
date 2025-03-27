import numpy as np
from ...circuit_utilities.circuit_errors import SymbQuantumStateError, SymbStatePreparationError
from ...circuit_config import *
import sympy as sp



class SymbQubit:
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'symbqubit')
        self.name: str = kwargs.get("name","|Quantum State>")
        self.rho = kwargs.get("rho", None)

    def __sub__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":
        """Subtraction of two SymbQubit rho matrices, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = self.rho - other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}             
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __add__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":
        """Addition of two SymbQubit rho matrices, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = self.rho + other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __matmul__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":    
        """Matrix multiplication between two SymbQubit objects, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = self.rho * other.rho
            kwargs = {"rho": new_rho}
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected type SymbQubit")
    
    def __mod__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":
        """Tensor product among two SymbQubit objects, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = sp.tensorproduct(self.rho, other.rho)
            kwargs = {"rho": new_rho}
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")