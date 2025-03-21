import numpy as np
from utilities.qc_errors import GateError
from .static_methods.gate_methods import *
from .qubit import combine_qubit_attr
from utilities.config import p_prec
from utilities.validation_funcs import gate_validation


def combine_gate_attr(self: "Gate", other: "Gate", op = "+") -> list:
        """Allows the returned objects to still return name too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            kwargs["name"] = f"{self.name} {op} {other.name}"
        return kwargs

class Gate:
    def __init__(self, **kwargs):
        self.class_type = "gate"
        self.name = kwargs.get("name", "Quantum Gate")
        self.matrix = kwargs.get("matrix", None)
        gate_validation(self)
        self.dim: int = len(self.matrix)
        self.length = self.dim ** 2
        self.n: int =  int(np.log2(self.dim))
        
    def __str__(self) -> str:
        matrix_str = np.array2string(self.matrix, precision=p_prec, separator=', ', suppress_small=True)
        return f"{self.name}\n{matrix_str}"
      
        
    def __and__(self, other) -> "Gate":
        if isinstance(other, Gate):
            new_matrix = np.block([[self.matrix, np.zeros_like(other.matrix)], [np.zeros_like(self.matrix), other.matrix]])
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "&"))
            return Gate(**kwargs)
        else:
            raise GateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Gate):
            if self.matrix is not None and other.matrix is not None:
                return self.matrix == other.matrix
            raise GateError(f"The inputted objects must have attr: self.matrix and other.matrix")
        raise GateError(f"Cannot have types {type(self)} and {type(other)}, expected two Gate classes")
    
    def __getitem__(self, index: tuple[int, int]) -> np.ndarray:
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.matrix[row, col]

    def __mod__(self, other) -> "Gate":            #tensor product
        if isinstance(other, Gate):
            new_matrix = np.kron(self.matrix, other.matrix)
            new_matrix = np.round(new_matrix, decimals=10)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "%"))
            return Gate(**kwargs)
        else:
            raise GateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
        
    def __mul__(self: "Gate", other: int | float) -> "Gate":
        if isinstance(other, (int, float)):
            new_mat = self.matrix * other
            kwargs = {"matrix": new_mat}
            kwargs.update(combine_gate_attr(self, other, op = "*"))
            return Gate(**kwargs)
        raise GateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")

    def __rmul__(self: "Gate", other: int | float) -> "Gate":
        return self.__mul__(other)
    
    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.matrix *= other
            return self
        raise GateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")
        
    def __matmul__(self, other) -> "Gate":
        if isinstance(other, Gate):
            new_matrix = np.dot(self.matrix, other.matrix)
            new_matrix = np.round(new_matrix, decimals=10)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return Gate(**kwargs)
        elif other.class_type == "qubit":
            new_rho = np.dot(np.dot(self.matrix, other.rho), np.conj(self.matrix.T))
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return other.__class__(**kwargs)
        elif isinstance(other, np.ndarray):
            new_matrix = np.dot(self.matrix, other)
            new_matrix = np.round(new_matrix, decimals=10)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return Gate(**kwargs)
        raise GateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")

    @classmethod                             #creates any specific control gate
    def C_Gate(cls, **kwargs):
        """The Control Gate, commonly seen in the form of the CNOT Gate, used to entangle Qubits
        Args:
            type: str: can either give "standard" type or "inverted" type
            gate: Gate: selects the gate action, eg X for CNOT, defaults to X_Gate
        Returns:
            Gate: The specified Control Gate"""
        gate_type: str = kwargs.get("type", "standard")
        gate_action: Gate = kwargs.get("gate", X_Gate)
        new_gate: Gate = Identity & gate_action
        if gate_type == "standard":
            return cls(name=f"Control {gate_action.name}", matrix=new_gate.matrix)
        elif gate_type == "inverted":
            new_gate = Gate.Swap() @ new_gate @ Gate.Swap()
            return cls(name=f"Inverted Control {gate_action.name}", matrix=new_gate.matrix)

    @classmethod
    def Identity(cls, **kwargs):
        return identity_gate(cls, **kwargs)

    @classmethod
    def Hadamard(cls):
        return Hadamard(cls)
    
    @classmethod
    def X_Gate(cls):
        return X_Gate(cls)
    
    @classmethod
    def Y_Gate(cls):
        return Y_Gate(cls)
    
    @classmethod
    def Z_Gate(cls):
        return Z_Gate(cls)
    
    @classmethod
    def X_Gate(cls):
        return X_Gate(cls)
    
    @classmethod
    def P_Gate(cls, theta, **kwargs):
        return phase_gate(cls, theta, **kwargs)
    
    @classmethod
    def U_Gate(cls, a, b, c):
        return unitary_gate(cls, a, b, c)
    
    @classmethod
    def Swap(cls, **kwargs):
        return swap_gate(cls, **kwargs)



X_Gate = Gate.X_Gate()             #initialises the default gates
Y_Gate = Gate.Y_Gate()
Z_Gate = Gate.Z_Gate()
Identity = Gate.Identity()
Hadamard = Gate.Hadamard()
CNot_flip = Gate.C_Gate(type="inverted", name="CNot_flip")
CNot = Gate.C_Gate(type="standard", name="CNot")
Swap = Gate.Swap()
S_Gate = Gate.P_Gate(theta=np.pi/2, name="S Gate")
T_Gate = Gate.P_Gate(theta=np.pi/4, name="T Gate")