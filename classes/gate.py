import numpy as np
from utilities.qc_errors import GateError
from .static_methods.gate_methods import *
from .qubit import combine_qubit_attr
from utilities.config import p_prec


def combine_gate_attr(self, other, op = "+"):
        """Allows the returned objects to still return name too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            kwargs["name"] = f"{self.name} {op} {other.name}"
        return kwargs

class Gate:
    def __init__(self, **kwargs):
        self.class_type = "gate"
        self.name = kwargs.get("name", None)
        self.matrix = kwargs.get("matrix", None)
        self.gate_validation()
        self.dim: int = len(self.matrix)
        self.length = self.dim ** 2
        self.n: int =  int(np.log2(self.dim))
        


    def gate_validation(self):
        if self.matrix is None:
            raise GateError(f"Gates can only be initialised if they are provided with a matrix")
        if not isinstance(self.matrix, (list, np.ndarray)):
            raise GateError(f"The gate cannot be of type: {type(self.matrix)}, expected type list or np.ndarray")
        self.matrix = np.array(self.matrix, dtype=np.complex128)
        if np.size(self.matrix) != 1:
            if self.matrix.shape[0] != self.matrix.shape[1]:
                raise GateError(f"All gates must be of a square shape. This gate has shape {self.matrix.shape[0]} x {self.matrix.shape[1]}")
            gate_check = np.dot(np.conj(self.matrix.T), self.matrix)
            if not np.all(np.isclose(np.diag(gate_check),1.0, atol=1e-3)):
                raise GateError(f"This gate is not unitary {self.matrix}")


    def __str__(self):
        matrix_str = np.array2string(self.matrix, precision=p_prec, separator=', ', suppress_small=True)
        if self.name is not None:
            return f"{self.name}\n{matrix_str}"
        else:
            return f"Quantum Gate\n{matrix_str}"
        
    def __and__(self, other):
        if isinstance(other, self.__class__):
            new_matrix = np.block([[self.matrix, np.zeros_like(other.matrix)], [np.zeros_like(self.matrix), other.matrix]])
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "&"))
            return self.__class__(**kwargs)
        else:
            raise GateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.matrix is not None and other.matrix is not None:
                return self.matrix == other.matrix
            raise GateError(f"The inputted objects must have attr: self.matrix and other.matrix")
        raise GateError(f"Cannot have types {type(self)} and {type(other)}, expected two Gate classes")
    
    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.matrix[row, col]

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            new_matrix = np.kron(self.matrix, other.matrix)
            new_matrix = np.round(new_matrix, decimals=10)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return self.__class__(**kwargs)
        else:
            raise GateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
        
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            new_matrix = np.dot(self.matrix, other.matrix)
            new_matrix = np.round(new_matrix, decimals=10)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "*"))
            return self.__class__(**kwargs)
        elif other.class_type == "qubit":
            new_rho = np.dot(np.dot(self.matrix, other.rho), np.conj(self.matrix.T))
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "*"))
            return other.__class__(**kwargs)
        elif isinstance(other, np.ndarray):
            new_matrix = np.dot(self.matrix, other)
            new_matrix = np.round(new_matrix, decimals=10)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "*"))
            return self.__class__(**kwargs)
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
            new_gate = Gate.Swap() * new_gate * Gate.Swap()
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