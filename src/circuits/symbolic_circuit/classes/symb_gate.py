import numpy as np
from ...base_classes.base_gate import *
from ..utilities.circuit_errors import SymbGateError
from ...circuit_config import *
import sympy as sp
from ..static_methods.symb_gate_methods import *


__all__ = ["SymbGate", "X_Gate_symb", "Y_Gate_symb", "Z_Gate_symb", "Identity_symb", "Hadamard_symb",
           "U_Gate_symb", "S_Gate_symb", "T_Gate_symb", "Swap_symb", "P_Gate_symb", "Rotation_x_symb",
           "Rotation_y_symb", "Rotation_z_symb"]

class SymbGate(BaseGate):
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'symbgate')
        super().__init__(**kwargs)
        self.dim: int = int(self.matrix.shape[0])
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
    
    def subs(self, substitution: dict) -> "SymbGate":
        self.matrix = self.matrix.subs(substitution)
        return self


    
    @classmethod
    def Identity(cls, **kwargs):
        gate = identity_gate(cls, **kwargs)
        return gate

    @classmethod
    def Hadamard(cls):
        gate = hadamard_gate(cls)
        return gate
    
    @classmethod
    def X_Gate(cls):
        gate = pauli_x_gate(cls)
        return gate
    
    @classmethod
    def Y_Gate(cls):
        gate = pauli_y_gate(cls)
        return gate
    
    @classmethod
    def Z_Gate(cls):
        gate = pauli_z_gate(cls)
        return gate
    
    @classmethod
    def P_Gate(cls, **kwargs):
        gate = phase_gate(cls, **kwargs)
        return gate
    
    @classmethod
    def U_Gate(cls, **kwargs):
        gate = unitary_gate(cls, **kwargs)
        return gate
    
    @classmethod
    def Swap(cls):
        gate = swap_gate(cls)
        return gate
    
    @classmethod
    def Rotation_X(cls, **kwargs):
        gate = rotation_x_gate(cls, **kwargs)
        return gate
    
    @classmethod
    def Rotation_Y(cls, **kwargs):
        gate = rotation_y_gate(cls, **kwargs)
        return gate
    
    @classmethod
    def Rotation_Z(cls, **kwargs):
        gate = rotation_z_gate(cls, **kwargs)
        return gate
    

X_Gate_symb = SymbGate.X_Gate()             #initialises the default gates
Y_Gate_symb = SymbGate.Y_Gate()
Z_Gate_symb = SymbGate.Z_Gate()
Identity_symb = SymbGate.Identity()
Hadamard_symb = SymbGate.Hadamard()
Swap_symb = SymbGate.Swap()
U_Gate_symb = SymbGate.U_Gate()
P_Gate_symb = SymbGate.P_Gate()
Rotation_x_symb = SymbGate.Rotation_X()
Rotation_y_symb = SymbGate.Rotation_Y()
Rotation_z_symb = SymbGate.Rotation_Z()
S_Gate_symb = SymbGate.P_Gate(theta=sp.pi/2, name="S Gate")
T_Gate_symb = SymbGate.P_Gate(theta=sp.pi/4, name="T Gate")