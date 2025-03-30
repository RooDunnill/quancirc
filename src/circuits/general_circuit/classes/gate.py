import numpy as np
from ...base_classes.base_gate import *
from ..utilities.circuit_errors import GateError
from ..static_methods.gate_methods import *
from .qubit import combine_qubit_attr
from ...circuit_config import *
from ...circuit_utilities.validation_funcs import gate_validation
from ...circuit_utilities.sparse_funcs import *


__all__ = ["Gate", "X_Gate", "Y_Gate", "Z_Gate", "Hadamard", "Identity", "CNot", "CNot_flip",
           "S_Gate", "T_Gate", "Swap"]

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

class Gate(BaseGate):
    all_immutable_attr = ["class_type"]
    immutable_attr = ["name", "matrix", "lenght", "n", "dim", "immutable"]
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'gate')
        super().__init__(**kwargs)
        gate_validation(self)
        self.dim: int = self.matrix.shape[0]
        self.length = self.dim ** 2
        self.n: int =  int(np.log2(self.dim))
        self.immutable = False
        
    @classmethod
    def __dir__(cls):
        return ["C_Gate", "Identity", "Hadamard", "Rotation_X", "Rotation_Y", "Rotation_Z", "X_Gate", "Y_Gate", "Z_Gate", "P_Gate", "U_Gate", "Swap"]

    def __dir__(self):
        methods = [None]
        return [func for func in methods if callable(getattr(self, func, None)) and not func.startswith("__")]

    def __str__(self) -> str:
        matrix_str = np.array2string(dense_mat(self.matrix), precision=p_prec, separator=', ', suppress_small=True)
        return f"{self.name}\n{matrix_str}"
      
    def __setattr__(self, name, value):
        if getattr(self, "immutable", False) and name in self.immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        if name in self.all_immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        super().__setattr__(name, value)

    def __and__(self, other) -> "Gate":
        mat_1 = convert_to_sparse(self.matrix)
        mat_2 = convert_to_sparse(other.matrix)
        if isinstance(other, Gate):
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = sparse.bmat([[mat_1, sparse.csr_matrix(mat_1.shape)], [sparse.csr_matrix(mat_2.shape), mat_2.matrix]])
            else:
                mat_1 = dense_mat(self.matrix)
                mat_2 = dense_mat(other.matrix)
                new_matrix = np.block([[mat_1, np.zeros_like(mat_2)], [np.zeros_like(mat_1), mat_2]])
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
    

    def __mod__(self, other) -> "Gate":            #tensor product
        mat_1 = convert_to_sparse(self.matrix)
        mat_2 = convert_to_sparse(other.matrix)
        if isinstance(other, Gate):
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = sparse.kron(mat_1, mat_2)
            else:
                mat_1 = dense_mat(self.matrix)
                mat_2 = dense_mat(other.matrix)
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
                mat_1 = dense_mat(self.matrix)
                mat_2 = dense_mat(other.matrix)
                new_matrix = np.dot(mat_1, mat_2)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return Gate(**kwargs)
        elif other.class_type == "qubit":
            rho_2 = convert_to_sparse(other.rho)
            if sparse.issparse(mat_1) and sparse.issparse(rho_2):
                temp_rho = mat_1.dot(rho_2)
                new_rho = temp_rho.dot(mat_1.T)
            else:
                mat_1 = dense_mat(self.matrix)
                rho_2 = dense_mat(other.rho)
                new_rho = np.dot(np.dot(mat_1, rho_2), np.conj(mat_1.T))
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return other.__class__(**kwargs)
        elif other.class_type == "lwqubit":
            vec_2 = convert_to_sparse_array(other.state)
            if sparse.issparse(mat_1) and sparse.issparse(vec_2):
                new_vec = mat_1 @ vec_2.reshape((-1, 1))
            else:
                mat_1 = dense_mat(self.matrix)
                vec_2 = dense_mat(other.state)
                new_vec = np.dot(mat_1, vec_2)
            kwargs = {"state":new_vec}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return other.__class__(**kwargs)
        elif isinstance(other, (sparse.sparray, np.ndarray)):
            mat_2 = other
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = mat_1.dot(mat_2)
            else:
                mat_1 = dense_mat(self.matrix)
                mat_2 = dense_mat(other)
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