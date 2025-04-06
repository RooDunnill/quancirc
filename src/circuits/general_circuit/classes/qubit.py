from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from ...base_classes.base_qubit import *
from ...circuit_utilities.sparse_funcs import *
from ..utilities.circuit_errors import QuantumStateError, StatePreparationError
from ..static_methods.qubit_methods import *
from ...circuit_config import *
from ..utilities.validation_funcs import qubit_validation, rho_validation
from ...base_classes.base_qubit import copy_qubit_attr, combine_qubit_attr

__all__ = ["Qubit", "q0", "q1", "qp", "qm", "qpi", "qmi"]

@log_all_methods
class Qubit(BaseQubit):                                           #creates the qubit class
    immutable_attr = ["skip_val", "state", "dim", "length", "n", "rho", "name", "state_type", "immutable"]
    """The class to define and initialise Qubits and Quantum States"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, 'class_type', 'qubit')
        self.weights: list = kwargs.get("weights", None)
        self.rho: list = kwargs.get("rho", None)
        self.state_type = None
        qubit_validation(self)
        self.rho_init()
        rho_validation(self)
        self.set_state_type()
        self.dim = self.rho.shape[0]
        self.length = self.dim ** 2
        self.n = int(np.log2(self.dim))
        self._initialised = True


    @classmethod
    def __dir__(cls):
        return ["q0", "q1", "qp", "qm", "qpi", "qmi", "create_mixed_state"]

    def __dir__(self):
        methods = ["debug", "partial_trace", "isolate_qubit", "set_display_mode", "norm"]
        return [func for func in methods if callable(getattr(self, func, None)) and not func.startswith("__")]

    def __str__(self):
        self.set_state_type()
        return super().__str__()

    def rho_init(self) -> None:
        """Builds and checks the rho attribute during __init__, returns type None"""
        if self.rho is None and self.state is None:
            self.rho = np.eye(1)
            self.skip_val = True
        if self.rho is None:
            if self.weights is not None:
                self.rho = self.build_mixed_rho()
            else:
                self.rho = self.build_pure_rho()

    def __setattr__(self: "Qubit", name: str, value) -> None:
        super().__setattr__(name, value)
        

    def __mod__(self: "Qubit", other: "Qubit") -> "Qubit":
        """Tensor product among two Qubit objects, returns a Qubit object"""
        if isinstance(other, Qubit):
            rho_1, rho_2 = auto_choose(self.rho, other.rho, tensor=True)
            if sparse.issparse(rho_1):
                new_rho = sparse.kron(rho_1, rho_2)
            else:
                new_rho = np.kron(rho_1, rho_2)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "%"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __matmul__(self: "Qubit", other: "Qubit") -> "Qubit":     
        """Matrix multiplication between two Qubit objects, returns a Qubit object"""
        if isinstance(other, Qubit):
            rho_1, rho_2 = auto_choose(self.rho, other.rho)
            if sparse.issparse(rho_1):
                new_rho = rho_1 @ rho_2   #swapped indice order for the transpose
            else:
                new_rho = np.dot(rho_1, rho_2)
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")
    
    def __or__(self: "Qubit", other: "Qubit") -> "Qubit":       #rho | gate
        """Non unitary matrix multiplication between a gate and a Qubit, used mostly for Quantum Information Calculations, returns a Qubit object"""
        if isinstance(other, Qubit):
            raise QuantumStateError(f"Cannot matrix multiply (singular) two Quantum states together")
        elif other.class_type == "gate":
            new_rho = np.dot(self.rho, other.matrix)
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho, "skip_validation": True}            #CAREFUL skip val here
            kwargs.update(combine_qubit_attr(self, other, op = "|"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")

    def __and__(self: "Qubit", other: "Qubit") -> "Qubit":
        """Direct sum of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, Qubit):
            self.rho = dense_mat(self.rho)
            other.rho = dense_mat(other.rho)
            new_rho = np.block([[self.rho, np.zeros_like(other.rho)], [np.zeros_like(self.rho), other.rho]])
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "&"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __eq__(self: "Qubit", other: "Qubit") -> bool:
        """Checks if two Qubit object rho matrices are identical, returns bool type"""
        if isinstance(other, Qubit):
            if self.rho is not None and other.rho is not None:
                self.rho = dense_mat(self.rho)
                other.rho = dense_mat(other.rho)
                return np.allclose(self.rho, other.rho, atol=1e-4)
            raise QuantumStateError(f"The inputted objects must have attr: self.rho and other.rho")
        raise QuantumStateError(f"Cannot have types {type(self)} and {type(other)}, expected two Qubit classes")
    
    def __getitem__(self: "Qubit", index: int | slice):
        """Allows for a single Qubit to be returned from an index of a multi Qubit state, returns Qubit object"""
        if isinstance(index, slice):
            start, stop, step = index.indices(self.n)
            selected_qubits = [start, stop - 1]
            get_qubit = self.isolate_qubit(selected_qubits)
            return get_qubit
        elif isinstance(index, int):
            get_qubit = self.isolate_qubit(index)
            if get_qubit is None:
                raise QuantumStateError(f"Could not isolate qubit {index}, invalid index input")
            get_qubit.index = index
            return get_qubit
        raise QuantumStateError(f"Index cannot be of type {type(index)}, expected type int or slice")
    
    def __setitem__(self: "Qubit", index: int | slice, new_state: "Qubit") -> None:
        """Sets a sinlge Qubit to the inputted Qubit, then tensors the state back into the multistate, returns None type"""
        if not isinstance(self.rho, (sparse.spmatrix, np.ndarray)):
            raise QuantumStateError(f"self.rho cannot be of type {type(self.rho)}, must be of type sp.spmatrix or type np.ndarray")
        if isinstance(index, int):
            rho_A, replaced_qubit, rho_B = self.isolate_qubit(index, con=True)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.n)
            selected_qubits = [start, stop - 1]
            rho_A, replaced_qubit, rho_B = self.isolate_qubit(selected_qubits, con=True)
        if replaced_qubit.dim == new_state.dim:
            rho_A.rho, replaced_qubit.rho, rho_B.rho = auto_choose(rho_A.rho, replaced_qubit.rho, rho_B.rho, tensor=True)
            new_state = rho_A % new_state % rho_B
            if new_state.dim == self.dim:
                self.rho = new_state.rho
            else:
                raise QuantumStateError(f"The new rho needs to be the same dimension as the old rho, not {self.dim} and {new_state.dim}")
        else:
            raise QuantumStateError(f"The dimensions of the new qubit must be the same as the dimensions of the old qubit, not {replaced_qubit.dim} and {new_state.dim}")

    def partial_trace(self: "Qubit", size_a: int, size_c: int, **kwargs) -> "Qubit":
        rho = kwargs.get("rho", self.rho)
        if not isinstance(rho, (sparse.spmatrix, np.ndarray, list)):
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
        kwargs.update(copy_qubit_attr(self))
        return Qubit(**kwargs)

    def isolate_qubit(self: "Qubit", qubit_index: int | list, con=False) -> Qubit | tuple[Qubit, Qubit, Qubit]:
        """Used primarily in __getitem__ to return a single Qubit from a multiqubit state, returns a Qubit object"""
        if qubit_index is None:
            raise QuantumStateError(f"Must have an inputted qubit value, expected type int or type list") 
        if qubit_index is not None:
            if isinstance(qubit_index, int):
                if qubit_index > self.n - 1:
                    raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
                if con is True:
                    A_rho = self.partial_trace(0, self.n - qubit_index)
                    B_rho = self.partial_trace(qubit_index + 1,0)
                    isolated_rho = self.partial_trace(qubit_index, self.n - qubit_index - 1)
                    return A_rho, isolated_rho, B_rho
                else:
                    return self.partial_trace(qubit_index, self.n - qubit_index - 1)
            elif isinstance(qubit_index, list) and len(qubit_index) == 2:
                if max(qubit_index) > self.n - 1:
                    raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
                if con is True:
                    A_rho = self.partial_trace(0, self.n - qubit_index[0])
                    B_rho = self.partial_trace(qubit_index[1] + 1,0)
                    isolated_rho = self.partial_trace(qubit_index[0], self.n - qubit_index[1] - 1)
                    return A_rho, isolated_rho, B_rho
                else:
                    return self.partial_trace(qubit_index[0], self.n - qubit_index[1] - 1)
            raise QuantumStateError(f"qubit_index cannot be of type {type(qubit_index)}, expected either type int or type list of length 2")
        raise QuantumStateError(f"Must provide a qubit_index value of type: int")
        
    def norm(self: "Qubit") -> None:
        """Normalises a rho matrix, returns type None"""
        if self.rho.shape[0] == self.rho.shape[1]:
            if sparse.issparse(self.rho):
                trace_rho = self.rho.diagonal().sum()
            else:
                trace_rho = np.trace(self.rho)
            if trace_rho != 0:
                self.rho = self.rho / trace_rho
            elif not self.skip_val:
                raise QuantumStateError(f"The trace of the density matrix cannot be 0 and so cannot normalise")
        else:
            raise QuantumStateError(f"self.rho must be a square matrix, not of shape {self.rho.shape}")
    
    def build_mixed_rho(self: "Qubit") -> np.ndarray:
        """Builds a mixed rho matrix, primarily in initiation of Qubit object, returns type np.ndarray"""
        if self.weights is not None:
            mixed_rho = np.zeros((len(self.state[0]),len(self.state[0])), dtype=np.complex128)
            for i in range(len(self.weights)):
                mixed_rho += self.weights[i] * np.einsum("j,k->jk", np.conj(self.state[i]), self.state[i], optimize=True)
            return mixed_rho
        raise StatePreparationError(f"For a mixed rho to be made, you must provide weights in kwargs")
        
    def build_state_from_rho(self: "Qubit") -> np.ndarray:
        """Builds a state vector from the given rho matrix, primarily for printing purposes, returns type np.ndarray"""
        if sparse.issparse(self.rho):
            if np.all(np.isclose(self.rho.data, 0.0, atol=1e-4)):
                return np.zeros(self.rho.shape[0], dtype=np.complex128)
            k = int(np.log(self.rho.shape[0]))
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
            raise QuantumStateError(f"You cannot have a probability over 1, the probabilities {probs} have been computed incorrectly")
        if np.any(np.isclose(probs, 1.0)):
            state_vector = states[:, max_eigenvalue]
            norm =  sparse.linalg.norm(state_vector) if sparse.issparse(state_vector) else np.linalg.norm(state_vector)
            if norm != 0:
                state_vector /= norm
                return state_vector
            raise QuantumStateError(f"The norm cannot be 0")
        return probs, states
    
    @classmethod
    def create_mixed_state(self: "Qubit", states: list, weights: list) -> "Qubit":
        """This is used for when you want to combine premade states into a larger mixed state"""
        if not isinstance(states, list) or not isinstance(weights, list):
            raise StatePreparationError(f"states and weights cannot be of type, {type(states)} and {type(weights)}, must be of type list and list")
        if not all(isinstance(state, Qubit) for state in states) or not all(isinstance(probs, float) for probs in weights):
            raise StatePreparationError(f"States and weights must be made up of types Qubit and types float")
        if len(states) != len(weights):
            raise StatePreparationError(f"The amount of states must match the amount of weights given, not {len(states)} and {len(weights)}")
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
        return Qubit(**kwargs)
        
    

    

    @classmethod
    def q0(cls, **kwargs):
        return q0_state(cls, **kwargs)

    @classmethod
    def q1(cls, **kwargs):
        return q1_state(cls, **kwargs)

    @classmethod
    def qp(cls):
        return qp_state(cls)

    @classmethod
    def qm(cls):
        return qm_state(cls)

    @classmethod
    def qpi(cls):
        return qpi_state(cls)

    @classmethod
    def qmi(cls):
        return qmi_state(cls)

q0 = Qubit.q0()
q0.immutable = True
q1 = Qubit.q1()
q1.immutable = True
qp = Qubit.qp()
qp.immutable = True
qm = Qubit.qm()
qm.immutable = True
qpi = Qubit.qpi()
qpi.immutable = True
qmi = Qubit.qmi()
qmi.immutable = True

