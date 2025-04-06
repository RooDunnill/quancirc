from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from ...base_classes.base_quant_state import *
from ..static_methods.qutrit_methods import *
from ...circuit_utilities.sparse_funcs import *
from ..utilities.validation_funcs import qutrit_validation, rho_validation
from ...circuit_config import *
from ..utilities.circuit_errors import QutritStateError, QutritStatePreparationError
from ...base_classes.base_quant_state import copy_quant_state_attr, combine_quant_state_attr
from ..utilities.gen_utilities import *



__all__ = ["Qutrit", "qt0", "qt1", "qt2"]




@log_all_methods
class Qutrit(BaseQuantState):                                     
    """The class to define and initialise Qutrits and Quantum States"""
    
    immutable_attr = ["skip_val", "state", "dim", "length", "n", "rho", "id", "state_type", "immutable", "print_history"]
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, 'class_type', 'Qutrit')
        self.weights: list = kwargs.get("weights", None)
        qutrit_validation(self)
        self.rho_init()
        rho_validation(self)
        self.set_state_type()
        self.dim = self.rho.shape[0]
        self.length = self.dim ** 2
        self.n = int(log_3(self.dim))
        self._initialised = True


    @classmethod
    def __dir__(cls):
        return ["q0", "q1", "qp", "qm", "qpi", "qmi", "create_mixed_state"]

    def __dir__(self):
        methods = ["debug", "partial_trace", "isolate_Qutrit", "set_display_mode", "norm"]
        return [func for func in methods if callable(getattr(self, func, None)) and not func.startswith("__")]

    def __str__(self):
        self.set_state_type()
        return super().__str__()

    def rho_init(self) -> None:
        """Builds and checks the rho attribute during __init__, returns type None"""
        if self.rho is None:
            if self.weights is not None:
                self.rho = self.build_mixed_rho()
            else:
                self.rho = self.build_pure_rho()

    def __setattr__(self: "Qutrit", name: str, value) -> None:
        super().__setattr__(name, value)
    
    def __or__(self: "Qutrit", other: "Qutrit") -> "Qutrit":       #rho | gate
        """Non unitary matrix multiplication between a gate and a Qutrit, used mostly for Quantum Information Calculations, returns a Qutrit object"""
        if isinstance(other, Qutrit):
            raise QutritStateError(f"Cannot matrix multiply (singular) two Quantum states together")
        elif other.class_type == "gate":
            new_rho = np.dot(self.rho, other.matrix)
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho, "skip_val": True}            #CAREFUL skip val here
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            return Qutrit(**kwargs)
        raise QutritStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qutrit or np.ndarray")

    def __and__(self: "Qutrit", other: "Qutrit") -> "Qutrit":
        """Direct sum of two Qutrit rho matrices, returns a Qutrit object"""
        if isinstance(other, Qutrit):
            self.rho = dense_mat(self.rho)
            other.rho = dense_mat(other.rho)
            new_rho = np.block([[self.rho, np.zeros_like(other.rho)], [np.zeros_like(self.rho), other.rho]])
            kwargs = {"rho": new_rho, "skip_val": True}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            kwargs["history"].append(f"Direct summed with State {other.id}") if "history" in kwargs else None
            return Qutrit(**kwargs)
        raise QutritStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __eq__(self: "Qutrit", other: "Qutrit") -> bool:
        """Checks if two Qutrit object rho matrices are identical, returns bool type"""
        if isinstance(other, Qutrit):
            if self.rho is not None and other.rho is not None:
                self.rho = dense_mat(self.rho)
                other.rho = dense_mat(other.rho)
                return np.allclose(self.rho, other.rho, atol=1e-4)
            raise QutritStateError(f"The inputted objects must have attr: self.rho and other.rho")
        raise QutritStateError(f"Cannot have types {type(self)} and {type(other)}, expected two Qutrit classes")
    
    def __getitem__(self: "Qutrit", index: int | slice):
        """Allows for a single Qutrit to be returned from an index of a multi Qutrit state, returns Qutrit object"""
        if isinstance(index, slice):
            start, stop, step = index.indices(self.n)
            selected_Qutrits = [start, stop - 1]
            get_Qutrit = self.isolate_Qutrit(selected_Qutrits)
            return get_Qutrit
        elif isinstance(index, int):
            get_Qutrit = self.isolate_Qutrit(index)
            if get_Qutrit is None:
                raise QutritStateError(f"Could not isolate Qutrit {index}, invalid index input")
            get_Qutrit.index = index
            return get_Qutrit
        raise QutritStateError(f"Index cannot be of type {type(index)}, expected type int or slice")
    
    def __setitem__(self: "Qutrit", index: int | slice, new_state: "Qutrit") -> None:
        """Sets a sinlge Qutrit to the inputted Qutrit, then tensors the state back into the multistate, returns None type"""
        if not isinstance(self.rho, (sparse.spmatrix, np.ndarray)):
            raise QutritStateError(f"self.rho cannot be of type {type(self.rho)}, must be of type sp.spmatrix or type np.ndarray")
        if isinstance(index, int):
            rho_A, replaced_Qutrit, rho_B = self.isolate_Qutrit(index, con=True)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.n)
            selected_Qutrits = [start, stop - 1]
            rho_A, replaced_Qutrit, rho_B = self.isolate_Qutrit(selected_Qutrits, con=True)
        if replaced_Qutrit.dim == new_state.dim:
            rho_A.rho, replaced_Qutrit.rho, rho_B.rho = auto_choose(rho_A.rho, replaced_Qutrit.rho, rho_B.rho, tensor=True)
            new_state = rho_A % new_state % rho_B
            if new_state.dim == self.dim:
                self.rho = new_state.rho
            else:
                raise QutritStateError(f"The new rho needs to be the same dimension as the old rho, not {self.dim} and {new_state.dim}")
        else:
            raise QutritStateError(f"The dimensions of the new Qutrit must be the same as the dimensions of the old Qutrit, not {replaced_Qutrit.dim} and {new_state.dim}")

    def partial_trace(self: "Qutrit", size_a: int, size_c: int, **kwargs) -> "Qutrit":
        rho = kwargs.get("rho", self.rho)
        if not isinstance(rho, (sparse.spmatrix, np.ndarray, list)):
            raise QutritStateError(f"rho cannot be of type {type(rho)}, expected type sp.spmatrix or type np.ndarray or type list")
        if not isinstance(size_a, int) and not isinstance(size_c, int):
            raise QutritStateError(f"size_a and size_c cannot be of types: {type(size_a)} and {type(size_c)}, expected types int and int")
        rho = dense_mat(rho)
        dim_a = int(3**size_a)
        dim_c = int(3**size_c)
        rho_dim = len(rho)
        dim_b = int(rho_dim/(dim_a*dim_c))
        if size_c == 0:
            new_rho = np.trace(rho.reshape(dim_a, dim_b, dim_a, dim_b), axis1=0, axis2=2)
        elif size_a == 0:
            new_rho = np.trace(rho.reshape(dim_b, dim_c, dim_b, dim_c), axis1=1, axis2=3)
        else:
            new_rho = np.trace(rho.reshape(dim_a, dim_b * dim_c, dim_a, dim_b * dim_c), axis1=0, axis2=2)
            new_rho = np.trace(new_rho.reshape(dim_b, dim_c, dim_b, dim_c), axis1=1, axis2=3)
        kwargs = {"rho": new_rho, "id": None, "history": [f"Traced out Qutrit of {self.id}"]}
        kwargs.update(copy_quant_state_attr(self, kwargs))
        return Qutrit(**kwargs)

    def isolate_Qutrit(self: "Qutrit", Qutrit_index: int | list, con=False) -> Qutrit | tuple[Qutrit, Qutrit, Qutrit]:
        """Used primarily in __getitem__ to return a single Qutrit from a multiQutrit state, returns a Qutrit object"""
        if Qutrit_index is None:
            raise QutritStateError(f"Must have an inputted Qutrit value, expected type int or type list") 
        if Qutrit_index is not None:
            if isinstance(Qutrit_index, int):
                if Qutrit_index > self.n - 1:
                    raise QutritStateError(f"The chosen Qutrit {Qutrit_index}, must be no more than the number of Qutrits in the state: {self.n}")
                if con is True:
                    A_rho = self.partial_trace(0, self.n - Qutrit_index)
                    B_rho = self.partial_trace(Qutrit_index + 1,0)
                    isolated_rho = self.partial_trace(Qutrit_index, self.n - Qutrit_index - 1)
                    return A_rho, isolated_rho, B_rho
                else:
                    return self.partial_trace(Qutrit_index, self.n - Qutrit_index - 1)
            elif isinstance(Qutrit_index, list) and len(Qutrit_index) == 2:
                if max(Qutrit_index) > self.n - 1:
                    raise QutritStateError(f"The chosen Qutrit {Qutrit_index}, must be no more than the number of Qutrits in the state: {self.n}")
                if con is True:
                    A_rho = self.partial_trace(0, self.n - Qutrit_index[0])
                    B_rho = self.partial_trace(Qutrit_index[1] + 1,0)
                    isolated_rho = self.partial_trace(Qutrit_index[0], self.n - Qutrit_index[1] - 1)
                    return A_rho, isolated_rho, B_rho
                else:
                    return self.partial_trace(Qutrit_index[0], self.n - Qutrit_index[1] - 1)
            raise QutritStateError(f"Qutrit_index cannot be of type {type(Qutrit_index)}, expected either type int or type list of length 2")
        raise QutritStateError(f"Must provide a Qutrit_index value of type: int")
        
    def norm(self: "Qutrit") -> None:
        """Normalises a rho matrix, returns type None"""
        if self.rho.shape[0] == self.rho.shape[1]:
            if sparse.issparse(self.rho):
                trace_rho = self.rho.diagonal().sum()
            else:
                trace_rho = np.trace(self.rho)
            if trace_rho != 0:
                self.rho = self.rho / trace_rho
            elif not self.skip_val:
                raise QutritStateError(f"The trace of the density matrix cannot be 0 and so cannot normalise")
        else:
            raise QutritStateError(f"self.rho must be a square matrix, not of shape {self.rho.shape}")
    
    def build_mixed_rho(self: "Qutrit") -> np.ndarray:
        """Builds a mixed rho matrix, primarily in initiation of Qutrit object, returns type np.ndarray"""
        if self.weights is not None:
            mixed_rho = np.zeros((len(self.state[0]),len(self.state[0])), dtype=np.complex128)
            for i in range(len(self.weights)):
                mixed_rho += self.weights[i] * np.einsum("j,k->jk", np.conj(self.state[i]), self.state[i], optimize=True)
            return mixed_rho
        raise QutritStatePreparationError(f"For a mixed rho to be made, you must provide weights in kwargs")
        
    def build_state_from_rho(self: "Qutrit") -> np.ndarray:
        """Builds a state vector from the given rho matrix, primarily for printing purposes, returns type np.ndarray"""
        if sparse.issparse(self.rho):
            if np.all(np.isclose(self.rho.data, 0.0, atol=1e-4)):
                return np.zeros(self.rho.shape[0], dtype=np.complex128)
            k = int(log_3(self.rho.shape[0]))
            probs, states = np.linalg.eig(dense_mat(self.rho)) if self.rho.shape[0] < eig_threshold else eigsh(self.rho, k=k, which="LM")
            probs = sparse.csr_matrix(probs, dtype=np.float64)
            states = sparse.csr_matrix(states, dtype=np.complex128)
        else:
            if np.all(np.isclose(self.rho, 0.0, atol=1e-4)):
                return np.zeros(self.rho.shape[0], dtype=np.complex128)
            probs, states = np.linalg.eigh(self.rho)
            probs = np.array(probs, dtype=np.float64)
            states = np.array(states, dtype=np.complex128)
        probs = dense_mat(probs)
        max_eigenvalue = np.argmax(np.isclose(probs, 1.0))
        if np.any(probs) > 1.0:
            raise QutritStateError(f"You cannot have a probability over 1, the probabilities {probs} have been computed incorrectly")
        if np.any(np.isclose(probs, 1.0)):
            state_vector = states[:, max_eigenvalue]
            norm =  sparse.linalg.norm(state_vector) if sparse.issparse(state_vector) else np.linalg.norm(state_vector)
            if norm != 0:
                state_vector /= norm
                return state_vector
            raise QutritStateError(f"The norm cannot be 0")
        return probs, states
    
    @classmethod
    def create_mixed_state(self: "Qutrit", states: list, weights: list) -> "Qutrit":
        """This is used for when you want to combine premade states into a larger mixed state"""
        if not isinstance(states, list) or not isinstance(weights, list):
            raise QutritStatePreparationError(f"states and weights cannot be of type, {type(states)} and {type(weights)}, must be of type list and list")
        if not all(isinstance(state, Qutrit) for state in states) or not all(isinstance(probs, float) for probs in weights):
            raise QutritStatePreparationError(f"States and weights must be made up of types Qutrit and types float")
        if len(states) != len(weights):
            raise QutritStatePreparationError(f"The amount of states must match the amount of weights given, not {len(states)} and {len(weights)}")
        all_sparse = all(sparse.issparse(state.rho) for state in states)
        if all_sparse:
            new_rho = sparse.csr_matrix((states[0].rho.shape[0], states[0].rho.shape[1]))
            for i, state in enumerate(states):
                new_rho  += weights[i] * sparse_mat(state.rho)
        else:
            for state in states:
                state.rho = dense_mat(state.rho)
            new_rho = np.tensordot(weights, [state.rho for state in states], axes=1)
        kwargs = {"rho": new_rho}
        return Qutrit(**kwargs)
        
    @classmethod
    def q0(cls, **kwargs):
        return q0_state(cls, **kwargs)

    @classmethod
    def q1(cls, **kwargs):
        return q1_state(cls, **kwargs)
    
    @classmethod
    def q2(cls, **kwargs):
        return q2_state(cls, **kwargs)
    

qt0 = Qutrit.q0()
qt0.immutable = True
qt1 = Qutrit.q1()
qt1.immutable = True
qt2 = Qutrit.q2()
qt2.immutable = True