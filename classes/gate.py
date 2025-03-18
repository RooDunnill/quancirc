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
        if self.matrix is None:
            raise GateError(f"Gates can only be initialised if they are provided with a matrix")
        if not isinstance(self.matrix, (list, np.ndarray)):
            raise GateError(f"The gate cannot be of type: {type(self.matrix)}, expected type list or np.ndarray")
        else:
            self.matrix = np.array(self.matrix, dtype=np.complex128)
            if self.matrix.shape[0] != self.matrix.shape[1]:
                raise GateError(f"All gates must be of a square shape. This gate has shape {self.matrix.shape[0]} x {self.matrix.shape[1]}")
        self.length: int = len(self.matrix)
        self.dim: int = int(np.sqrt(self.length))
        self.n: int =  0 if self.dim == 0 else int(np.log2(self.dim))

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
            new_rho = np.dot(np.dot(np.conj(self.matrix), other.rho), self.matrix)
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
    
    @staticmethod
    def FWHT(other):   #not currently working
        """The Fast Walsh Hadamard Transform, used heavily in Grover's to apply the tensored Hadamard"""
        if other.class_type == "qubit":
            sqrt2_inv = 1/np.sqrt(2)
            vec = other.state
            for i in range(other.n):                                            #loops through each size of qubit below the size of the state
                step_size = 2**(i + 1)                                          #is the dim of the current qubit tieration size 
                half_step = step_size // 2                                      #half the step size to go between odd indexes
                outer_range = np.arange(0, other.dim, step_size)[:, None]       #more efficient intergration of a loop over the state dim in steps of the current dim 
                inner_range = np.arange(half_step)                               
                indices = outer_range + inner_range                        
                a, b = vec[indices], vec[indices + half_step]
                vec[indices] = (a + b) * sqrt2_inv
                vec[indices + half_step] = (a - b) * sqrt2_inv                        #normalisation has been taken out giving a slight speed up in performance
            kwargs = {"state": vec}
            return other.__class__(**kwargs)
        raise TypeError(f"This can't act on this type, only on Qubits")


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