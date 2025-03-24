import numpy as np
from ..circuit_utilities.circuit_errors import GateError
from .static_methods.gate_methods import *
from .qubit import combine_qubit_attr
from ..circuit_config import *
from ..circuit_utilities.validation_funcs import gate_validation
from ..circuit_utilities.sparse_funcs import *

def combine_gate_attr(self: "Gate", other: "Gate", op = "+") -> list:
        """Allows the returned objects to still return name too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            kwargs["name"] = f"{self.name} {op} {other.name}"
        if hasattr(self, "skip_val") and self.skip_val == True:
            kwargs["skip_validation"] = True
        elif hasattr(other, "skip_val") and other.skip_val == True: 
            kwargs["skip_validation"] = True
        return kwargs

class Gate:
    all_immutable_attr = ["class_type"]
    immutable_attr = ["name", "matrix", "lenght", "n", "dim", "immutable"]
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'gate')
        self.skip_val = kwargs.get("skip_validation", False)
        self.name = kwargs.get("name", "Quantum Gate")
        self.matrix = kwargs.get("matrix", None)
        gate_validation(self)
        self.dim: int = len(self.matrix)
        self.length = self.dim ** 2
        self.n: int =  int(np.log2(self.dim))
        self.immutable = False
        
    def __str__(self) -> str:
        matrix_str = np.array2string(self.matrix, precision=p_prec, separator=', ', suppress_small=True)
        return f"{self.name}\n{matrix_str}"
      
    def __setattr__(self, name, value):
        if getattr(self, "immutable", False) and name in self.immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        if name in self.all_immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        super().__setattr__(name, value)

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
        mat_1 = convert_to_sparse(self.matrix)
        mat_2 = convert_to_sparse(other.matrix)
        if isinstance(other, Gate):
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = sparse.kron(mat_1, mat_2)
            else:
                new_matrix = np.kron(self.matrix, other.matrix)
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
            new_mat = self.matrix * other
            kwargs = {"matrix": new_mat}
            kwargs.update(combine_gate_attr(self, other, op = "*"))
            return Gate(**kwargs)
        raise GateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")
        
    def __matmul__(self, other) -> "Gate":
        mat_1 = convert_to_sparse(self.matrix)
        if isinstance(other, Gate):
            mat_2 = convert_to_sparse(other.matrix)
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_rho = mat_1.dot(mat_2)
            else:
                new_matrix = np.dot(mat_1, mat_2)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return Gate(**kwargs)
        elif other.class_type == "qubit":
            rho_2 = convert_to_sparse(other.rho)
            if sparse.issparse(mat_1) and sparse.issparse(rho_2):
                new_rho = mat_1.dot(rho_2)
            else:
                new_rho = np.dot(np.dot(mat_1, rho_2), np.conj(mat_1.T))
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return other.__class__(**kwargs)
        elif isinstance(other, (sparse.sparray, np.ndarray)):
            mat_2 = other
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = mat_1.dot(mat_2)
            else:
                new_matrix = np.dot(mat_1, mat_2)
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
        gate = identity_gate(cls, **kwargs)
        return gate

    @classmethod
    def Hadamard(cls):
        gate = hadamard_gate(cls)
        return gate
    
    @classmethod
    def Rotation_X(cls, theta):
        gate = rotation_x_gate(cls, theta)
        return gate
    
    @classmethod
    def Rotation_Y(cls, theta):
        gate = rotation_y_gate(cls, theta)
        return gate
    
    @classmethod
    def Rotation_Z(cls, theta):
        gate = rotation_z_gate(cls, theta)
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
    def P_Gate(cls, theta, **kwargs):
        gate = phase_gate(cls, theta, **kwargs)
        return gate
    
    @classmethod
    def U_Gate(cls, a, b, c):
        gate = unitary_gate(cls, a, b, c)
        return gate
    
    @classmethod
    def Swap(cls, **kwargs):
        gate = swap_gate(cls, **kwargs)
        return gate

    


X_Gate = Gate.X_Gate()             #initialises the default gates
X_Gate.immutable = True
Y_Gate = Gate.Y_Gate()
Y_Gate.immutable = True
Z_Gate = Gate.Z_Gate()
Z_Gate.immutable = True
Identity = Gate.Identity()
Identity.immutable = True
Hadamard = Gate.Hadamard()
Hadamard.immutable = True
CNot_flip = Gate.C_Gate(type="inverted", name="CNot_flip")
CNot_flip.immutable = True
CNot = Gate.C_Gate(type="standard", name="CNot")
CNot.immutable = True
Swap = Gate.Swap()
Swap.immutable = True
S_Gate = Gate.P_Gate(theta=np.pi/2, name="S Gate")
S_Gate.immutable = True
T_Gate = Gate.P_Gate(theta=np.pi/4, name="T Gate")
T_Gate.immutable = True