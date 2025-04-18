import numpy as np
from ...base_classes.base_quant_state import *
from ...general_circuit.classes.qubit import *
from ...general_circuit.classes.qubit import combine_quant_state_attr
from ...circuit_utilities.circuit_errors import *
from ..utilities.validation_funcs import lw_qubit_validation
from ..static_methods.qubit_methods import *
from ..utilities.circuit_errors import LwQuantumStateError, LwStatePreparationError
from ...circuit_utilities.sparse_funcs import *
from ...circuit_config import *



__all__ = ["LwQubit", "q0_lw", "q1_lw", "qp_lw", "qm_lw", "qpi_lw", "qmi_lw"]

class LwQubit: 
    """A lightweight variant of the Qubit class, which utilises 1D arrays and sparse matrices to make a faster quantum state
        at the forfeit of always being pure"""
    def __init__(self, **kwargs):
        object.__setattr__(self, "class_type", "lwqubit")
        object.__setattr__(self, "state_type", "Pure")
        self.skip_val = kwargs.get("skip_validation", False)
        self.display_mode = kwargs.get("display_mode", "density")
        self.name: str = kwargs.get("name","|\u03C8>")
        self.state = kwargs.get("state", None)
        self.index = None
        self.matrix_type = kwargs.get("matrix_type", "dense")
        self.index = kwargs.get("index", None)
        lw_qubit_validation(self)
        self.dim = self.state.shape[0]
        self.n = int(np.log2(self.dim))

    @classmethod
    def __dir__(cls):
        return ["q0_lw", "q1_lw", "qp_lw", "qm_lw", "qpi_lw", "qmi_lw"]
    
    def __dir__(self):
        return None
    
    def __matmul__(self: "LwQubit", other: "LwQubit") -> "LwQubit":
        raise LwQuantumStateError(f"Cannot matrix multiply two Lw Quantum states together togther")

    def __mod__(self: "LwQubit", other: "LwQubit") -> "LwQubit":
        """Tensor product among two Qubit objects, returns a Qubit object"""
        if isinstance(other, LwQubit):
            vec_1 = convert_to_sparse(self.state)
            vec_2 = convert_to_sparse(other.state)
            if sparse.issparse(vec_1) and sparse.issparse(vec_2):
                new_vec = sparse.kron(vec_1, vec_2)
            else:
                vec_1 = self.state
                vec_2 = other.state
                new_vec = np.kron(vec_1, vec_2)
            kwargs = {"state": new_vec}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            return LwQubit(**kwargs)
        raise LwQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __mul__(self: "LwQubit", other: int | float) -> "LwQubit":
        if isinstance(other, (int, float)):
            new_state = self.state * other
            kwargs = {"state": new_state, "skip_validation": True}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            return self.__class__(**kwargs)
        raise LwQuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __imul__(self: "LwQubit", other: float) -> "LwQubit":
        if isinstance(other, (int, float)):
            self.state *= other
            return self
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
    def __truediv__(self: "LwQubit", other: int | float) -> "LwQubit":
        if isinstance(other, (int, float)):
            new_state = self.state / other
            kwargs = {"state": new_state, "skip_validation": True}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __itruediv__(self: "LwQubit", other: float) -> "LwQubit":
        if isinstance(other, (int, float)):
            self.state /= other
            return self
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
    def __sub__(self: "LwQubit", other: "LwQubit") -> "LwQubit":
        """Subtraction of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, LwQubit):
            new_state = self.state - other.state
            kwargs = {"state": new_state, "skip_validation": True}                #CAREFUL skip val here
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            return self.__class__(**kwargs)
        raise LwQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __isub__(self: "LwQubit", other: "LwQubit") -> "LwQubit":
        if isinstance(other, LwQubit):
            if hasattr(self, "immutable"):
                raise LwQuantumStateError(f"This operation is not valid for an immutable object")
            self = self - other
            return self
        raise LwQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __add__(self: "LwQubit", other: "LwQubit") -> "LwQubit":
        """Addition of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, LwQubit):
            new_state = self.state + other.state
            kwargs = {"state": new_state, "skip_validation": True}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            return self.__class__(**kwargs)
        raise LwQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __iadd__(self: "LwQubit", other: "LwQubit") -> "LwQubit":
        if isinstance(other, LwQubit):
            if hasattr(self, "immutable"):
                raise LwQuantumStateError(f"This operation is not valid for an immutable object")
            self = self + other
            return self
        raise LwQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    """
    def partial_trace(self, size_a, size_c, **kwargs):
        state = kwargs.get("state", self.state)
        if not isinstance(state, (sparse.spmatrix, np.ndarray, list)):
            raise QuantumStateError(f"rho cannot be of type {type(rho)}, expected type sp.spmatrix or type np.ndarray or type list")
        if not isinstance(size_a, int) and not isinstance(size_c, int):
            raise QuantumStateError(f"size_a and size_c cannot be of types: {type(size_a)} and {type(size_c)}, expected types int and int")
        rho = dense_mat(rho)
        dim_a = int(2**size_a)
        dim_c = int(2**size_c)
        rho_dim = len(rho)
        dim_b = int(rho_dim/(dim_a*dim_c))
        if size_c == 0:
            new_rho = np.trace(rho.reshape(dim_a, dim_b, dim_a, dim_b), axis1=0, axis2=2)
        elif size_a == 0:
            new_rho = np.trace(rho.reshape(dim_b, dim_c, dim_b, dim_c), axis1=1, axis2=3)
        else:
            new_rho = np.trace(rho.reshape(dim_a, dim_b * dim_c, dim_a, dim_b * dim_c), axis1=0, axis2=2)
            new_rho = np.trace(new_rho.reshape(dim_b, dim_c, dim_b, dim_c), axis1=1, axis2=3)
        kwargs = {"rho": new_rho}
        kwargs.update(copy_quant_state_attr(self))
        return Qubit(**kwargs)
    """
    

    
    @classmethod
    def q0_lw(cls, **kwargs):
        return q0_state(cls, **kwargs)

    @classmethod
    def q1_lw(cls, **kwargs):
        return q1_state(cls, **kwargs)

    @classmethod
    def qp_lw(cls):
        return qp_state(cls)

    @classmethod
    def qm_lw(cls):
        return qm_state(cls)

    @classmethod
    def qpi_lw(cls):
        return qpi_state(cls)

    @classmethod
    def qmi_lw(cls):
        return qmi_state(cls)
    
    
    
q0_lw = LwQubit.q0_lw()
q0_lw.immutable = True
q1_lw = LwQubit.q1_lw()
q1_lw.immutable = True
qp_lw = LwQubit.qp_lw()
qp_lw.immutable = True
qm_lw = LwQubit.qm_lw()
qm_lw.immutable = True
qpi_lw = LwQubit.qpi_lw()
qpi_lw.immutable = True
qmi_lw = LwQubit.qmi_lw()
qmi_lw.immutable = True