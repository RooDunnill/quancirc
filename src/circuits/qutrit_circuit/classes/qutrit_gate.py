import numpy as np
from ...base_classes.base_gate import *
from ..utilities.circuit_errors import QutritGateError
from ..static_methods.gate_methods import *
from ...base_classes.base_quant_state import combine_quant_state_attr
from ...circuit_config import *
from ..utilities.validation_funcs import gate_validation
from ...circuit_utilities.sparse_funcs import *
from ..utilities.gen_utilities import *

__all__ = ["QutritGate"]


@log_all_methods
class QutritGate(BaseGate):
    all_immutable_attr = ["class_type"]
    immutable_attr = ["name", "matrix", "lenght", "n", "dim", "immutable"]
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'gate')
        super().__init__(**kwargs)
        gate_validation(self)
        self.dim: int = self.matrix.shape[0]
        self.length = self.dim ** 2
        self.n: int =  int(log_3(self.dim))
        self._initialised = True
        
    @classmethod
    def __dir__(cls):
        return []

    def __dir__(self):
        methods = [None]
        return [func for func in methods if callable(getattr(self, func, None)) and not func.startswith("__")]

    def __setattr__(self: "QutritGate", name: str, value) -> None:
        super().__setattr__(name, value)
      
    def __and__(self, other) -> "QutritGate":
        mat_1 = convert_to_sparse(self.matrix)
        mat_2 = convert_to_sparse(other.matrix)
        if isinstance(other, QutritGate):
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = sparse.bmat([[mat_1, sparse.csr_matrix(mat_1.shape)], [sparse.csr_matrix(mat_2.shape), mat_2.matrix]])
            else:
                mat_1 = dense_mat(self.matrix)
                mat_2 = dense_mat(other.matrix)
                new_matrix = np.block([[mat_1, np.zeros_like(mat_2)], [np.zeros_like(mat_1), mat_2]])
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "&"))
            return QutritGate(**kwargs)
        else:
            raise QutritGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __eq__(self, other) -> bool:
        if isinstance(other, QutritGate):
            if self.matrix is not None and other.matrix is not None:
                return self.matrix == other.matrix
            raise QutritGateError(f"The inputted objects must have attr: self.matrix and other.matrix")
        raise QutritGateError(f"Cannot have types {type(self)} and {type(other)}, expected two Gate classes")
    
    def __mod__(self, other) -> "QutritGate":            #tensor product
        if isinstance(other, QutritGate):
            mat_1, mat_2 = auto_choose(self.matrix, other.matrix, tensor=True)
            if sparse.issparse(mat_1):
                new_matrix = sparse.kron(mat_1 , mat_2)
            else:
                new_matrix = np.kron(mat_1, mat_2)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "%"))
            return QutritGate(**kwargs)
        else:
            raise QutritGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
        
    def __matmul__(self, other) -> "QutritGate":
        if isinstance(other, QutritGate):
            mat_1, mat_2 = auto_choose(self.matrix, other.matrix)
            if sparse.issparse(mat_1):
                new_matrix = mat_1 @ mat_2
            else:
                new_matrix = np.dot(mat_1, mat_2)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return QutritGate(**kwargs)
        elif isinstance(other, (sparse.sparray, np.ndarray)):
            mat_1, mat_2 = auto_choose(self.matrix, other)
            if sparse.issparse(mat_1):
                new_matrix = mat_1 @ mat_2
            else:
                new_matrix = np.dot(mat_1, mat_2)
            kwargs = {"matrix": new_matrix}
            kwargs.update(combine_gate_attr(self, other, op = "@"))
            return QutritGate(**kwargs)
        elif other.class_type == "qutrit":
            mat_1, rho_2 = auto_choose(self.matrix, other.rho)
            if sparse.issparse(mat_1):
                new_rho = mat_1 @ rho_2 @ mat_1.conj().T
            else:
                new_rho = np.dot(np.dot(mat_1, rho_2), np.conj(mat_1.T))
            kwargs = {"rho": new_rho}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            if "history" in kwargs:
                kwargs["history"].append(f"Applied {self.name} Gate")
            return other.__class__(**kwargs)
        raise QutritGateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")


    @classmethod
    def Identity(cls, **kwargs):
        gate = identity_gate(cls, **kwargs)
        return gate
    
    
    @classmethod
    def gm1(cls, theta, **kwargs):
        gate = gell_mann_1(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm2(cls, theta, **kwargs):
        gate = gell_mann_2(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm3(cls, theta, **kwargs):
        gate = gell_mann_3(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm4(cls, theta, **kwargs):
        gate = gell_mann_4(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm5(cls, theta, **kwargs):
        gate = gell_mann_5(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm6(cls, theta, **kwargs):
        gate = gell_mann_6(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm7(cls, theta, **kwargs):
        gate = gell_mann_7(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def gm8(cls, theta, **kwargs):
        gate = gell_mann_8(cls, theta=theta, **kwargs)
        return gate
    
    @classmethod
    def P_Gate(cls, theta, phi, gamma, **kwargs):
        gate = phase_gate(cls, theta, **kwargs)
        return gate
    
    @classmethod
    def H_uniform(cls):
        gate = hadamard_uniform(cls)
        return gate
    
    @classmethod
    def H_dft(cls):
        gate = hadamard_dft(cls)
        return gate
    
    def H_phase(cls, theta, **kwargs):
        gate = hadamard_phase(cls, theta, **kwargs)
        return gate
    
TriId = QutritGate.Identity()
TriId.immutable = True
Huni = QutritGate.H_uniform()
Huni.immutable = True 
Hdft = QutritGate.H_dft()
Hdft.immutable = True
