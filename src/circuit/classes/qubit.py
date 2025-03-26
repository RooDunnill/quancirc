import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import eigsh
from ..circuit_utilities.sparse_funcs import *
from ..circuit_utilities.circuit_errors import QuantumStateError, StatePreparationError, LWQuantumStateError
from .static_methods.qubit_methods import *
from ..circuit_config import *
from ..circuit_utilities.validation_funcs import qubit_validation, rho_validation


__all__ = ["Qubit", "q0", "q1", "qp", "qm", "qpi", "qmi"]

def combine_qubit_attr(self, other, op: str = None):
        """Allows the returned objects to still return name and info too"""
        kwargs = {}
        if op == "%":
            self_name_size = int(np.log2(self.dim))
            other_name_size = int(np.log2(other.dim)) 
            kwargs["name"] = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>"
        elif hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            if op:
                kwargs["name"] = f"{self.name} {op} {other.name}"
            else:
                kwargs["name"] = f"{self.name}"
        if isinstance(self, Qubit) and isinstance(other, Qubit):
            if isinstance(self.index, int) != isinstance(other.index, int):
                if isinstance(self.index, int):
                    kwargs["index"] = self.index
                else:
                    kwargs["index"] = other.index
            if hasattr(self, "display_mode") and hasattr(other, "display_mode"):
                if self.display_mode == "both" or other.display_mode == "both":
                    kwargs["display_mode"] = "both"
                elif self.display_mode == "density" or other.display_mode == "density":
                    kwargs["display_mode"] = "density"
                else:
                    kwargs["display_mode"] = "vector"
        elif isinstance(other, Qubit):
            if hasattr(other, "index"):
                kwargs["index"] = other.index
        if hasattr(self, "skip_val") and self.skip_val == True:
            kwargs["skip_validation"] = True
        elif hasattr(other, "skip_val") and other.skip_val == True: 
            kwargs["skip_validation"] = True
        return kwargs

def copy_qubit_attr(self):
    kwargs = {}
    if hasattr(self, "name"):
        kwargs["name"] = self.name
    if hasattr(self, "display_mode"):
        kwargs["display_mode"] = self.display_mode
    if hasattr(self, "skip_val") and self.skip_val == True:
        kwargs["skip_validation"] = True
    if hasattr(self, "index"):
        kwargs["index"] = self.index
    return kwargs

class Qubit:                                           #creates the qubit class
    all_immutable_attr = ["class_type"]
    immutable_attr = ["state", "dim", "length", "n", "rho", "name", "state_type", "immutable"]
    """The class to define and initialise Qubits and Quantum States"""
    def __init__(self, *args, **kwargs) -> None:
        object.__setattr__(self, 'class_type', 'qubit')
        self.skip_val = kwargs.get("skip_validation", False)
        self.matrix_type = kwargs.get("matrix_type", "dense")
        self.display_mode = kwargs.get("display_mode", "density")
        self.weights: list = kwargs.get("weights", None)
        self.name: str = kwargs.get("name","|Quantum State>")
        self.state: list = kwargs.get("state", None)
        self.rho: list = kwargs.get("rho", None)
        self.state_type = None
        self.index = kwargs.get("index", None)
        qubit_validation(self)
        self.rho_init()
        rho_validation(self)
        self.set_state_type()
        self.dim = len(dense_mat(self.rho))
        self.length = self.dim ** 2
        self.n = int(np.log2(self.dim))


    @classmethod
    def __dir__(cls):
        return ["q0", "q1", "qp", "qm", "qpi", "qmi"]

    def __dir__(self):
        methods = ["debug", "partial_trace", "isolate_qubit", "decompose_qubit", "set_display_mode", "norm"]
        return [func for func in methods if callable(getattr(self, func, None)) and not func.startswith("__")]

    def set_state_type(self) -> None:
        """Checks that state type and corrects if needed, returns type None"""
        purity_rho = dense_mat(self.rho)
        purity = np.trace(np.dot(purity_rho, purity_rho)).real
        if self.skip_val:
            self.state_type = "non unitary"
        elif np.isclose(purity, 1.0, atol=1e-4):
            self.state_type = "pure"
        elif purity < 1:
            self.state_type = "mixed"
        else:
            raise StatePreparationError(f"The purity of a state must be between 0 and 1, purity: {purity}")

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

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        state_print = self.build_state_from_rho()
        rho_str = np.array2string(dense_mat(self.rho), precision=p_prec, separator=', ', suppress_small=True)
        if self.state_type == "pure":
            if isinstance(state_print, tuple):
                raise StatePreparationError(f"The state vector of a pure state cannot be a tuple")
            state_str = np.array2string(dense_mat(state_print), precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"Pure Quantum State Vector:\n{state_str}"
            elif self.display_mode == "density":
                return f"Pure Quantum State Density Matrix:\n{rho_str}"
            elif self.display_mode == "both":
                return f"Pure Quantum State Vector:\n{state_str}\n\nDensity Matrix:\n{rho_str}"

        elif self.state_type == "mixed":
            if isinstance(state_print, np.ndarray):
                raise StatePreparationError(f"The state vector of a mixed state cannot be a sinlge np.ndarray")
            weights = dense_mat(state_print[0])
            state = dense_mat(state_print[1])
            weights_str = np.array2string(weights, precision=p_prec, separator=', ', suppress_small=True)
            state_str = np.array2string(state, precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"Mixed Quantum State Vector:\nWeights:\n{weights_str}\n\nStates:\n{state_str}"
            elif self.display_mode == "density":
                return f"Mixed Quantum State Density Matrix:\n{rho_str}"
            elif self.display_mode == "both":
                return f"Mixed Quantum State Vector:\nWeights:\n{weights_str}\n\nStates:\n{state_str}\n\nDensity Matrix:\n{rho_str}"
            
        elif self.state_type == "non unitary":
            return f"Non Quantum State Density Matrix:\n{rho_str}"
        
    def __setattr__(self, name, value):
        if getattr(self, "immutable", False) and name in self.immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        if name in self.all_immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        super().__setattr__(name, value)

    def __mod__(self: "Qubit", other: "Qubit") -> "Qubit":
        """Tensor product among two Qubit objects, returns a Qubit object"""
        if isinstance(other, Qubit):
            rho_1 = convert_to_sparse(self.rho)
            rho_2 = convert_to_sparse(other.rho)
            if sparse.issparse(rho_1) and sparse.issparse(rho_2):
                new_rho = sparse.kron(rho_1, rho_2)
            else:
                rho_1 = self.rho
                rho_2 = other.rho
                new_rho = np.kron(rho_1, rho_2)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "%"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
        
    def __matmul__(self: "Qubit", other: "Qubit") -> "Qubit":      #gateT @ rho @ gate
        """Matrix multiplication between two Qubit objects, returns a Qubit object"""
        if self.class_type == "qubit_lw":
            raise LWQuantumStateError(f"Lightweight States cannot be matrix multiplied with other quantum states")
        if isinstance(other, Qubit):
            raise QuantumStateError(f"Cannot matrix multiply (double) two Quantum states together")
        elif other.class_type == "gate":
            rho_1 = convert_to_sparse(self.rho)
            rho_2 = convert_to_sparse(other.rho)
            if sparse.issparse(rho_1) and sparse.issparse(rho_2):
                new_rho = rho_1.dot(rho_2)   #swapped indice order for the transpose
            else:
                new_rho = np.dot(rho_1, rho_2)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "*"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")
    
    def __mul__(self: "Qubit", other: int | float) -> "Qubit":
        if isinstance(other, (int, float)):
            new_rho = self.rho * other
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "*"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __rmul__(self: "Qubit", other: int | float) -> "Qubit":
        return self.__mul__(other)
    
    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.rho *= other
            return self
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
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

    def __sub__(self: "Qubit", other: "Qubit") -> "Qubit":
        """Subtraction of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, Qubit):
            new_rho = self.rho - other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}                #CAREFUL skip val here
            kwargs.update(combine_qubit_attr(self, other, op = "-"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __add__(self: "Qubit", other: "Qubit") -> "Qubit":
        """Addition of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, Qubit):
            new_rho = self.rho + other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "+"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __iadd__(self, other):
        if isinstance(other, Qubit):
            self.rho = self.rho + other.rho
            return self
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __and__(self: "Qubit", other: "Qubit") -> "Qubit":
        """Direct sum of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, Qubit):
            new_rho = np.block([[self.rho, np.zeros_like(other.rho)], [np.zeros_like(self.rho), other.rho]])
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "&"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __eq__(self: "Qubit", other: "Qubit") -> bool:
        """Checks if two Qubit object rho matrices are identical, returns bool type"""
        if isinstance(other, Qubit):
            if self.rho is not None and other.rho is not None:
                return np.allclose(self.rho, other.rho, atol=1e-4)
            raise QuantumStateError(f"The inputted objects must have attr: self.rho and other.rho")
        raise QuantumStateError(f"Cannot have types {type(self)} and {type(other)}, expected two Qubit classes")
    
    def __getitem__(self: "Qubit", index: int | slice):
        """Allows for a single Qubit to be returned from an index of a multi Qubit state, returns Qubit object"""
        if isinstance(index, slice):
            return [self[i] for i in range(self.n)]
        elif isinstance(index, int):
            get_qubit = self.isolate_qubit(index)
            if get_qubit is None:
                raise QuantumStateError(f"Could not isolate qubit {index}, invalid index input")
            get_qubit.index = index
            return get_qubit
        raise QuantumStateError(f"Index cannot be of type {type(index)}, expected type int or slice")
    
    def __setitem__(self: "Qubit", index: int, new_state: "Qubit") -> None:
        """Sets a sinlge Qubit to the inputted Qubit, then tensors the state back into the multistate, returns None type"""
        if not isinstance(self.rho, (sparse.spmatrix, np.ndarray)):
            raise QuantumStateError(f"self.rho cannot be of type {type(self.rho)}, must be of type sp.spmatrix or type np.ndarray")
        rho_A, replaced_qubit, rho_B = self.decompose_state(index)
        if replaced_qubit.dim == new_state.dim:
            new_state = rho_A % new_state % rho_B
            if new_state.dim == self.dim:
                self.rho = new_state.rho
            else:
                raise QuantumStateError(f"The new rho needs to be the same dimension as the old rho, not {self.dim} and {new_state.dim}")
        else:
            raise QuantumStateError(f"The dimensions of the new qubit must be the same as the dimensions of the old qubit, not {replaced_qubit.dim} and {new_state.dim}")

    def partial_trace(self, size_a, size_c, **kwargs):
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

    def isolate_qubit(self, qubit_index: int) -> "Qubit":
        """Used primarily in __getitem__ to return a single Qubit from a multiqubit state, returns a Qubit object"""
        if qubit_index is not None:
            if qubit_index > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
            if isinstance(qubit_index, int):
                isolated_rho = self.partial_trace(qubit_index, self.n - qubit_index - 1)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit_index)}, expected int") 
            return isolated_rho
        raise QuantumStateError(f"Must provide a qubit_index value of type: int")
        
    def decompose_state(self, qubit_index: int) -> tuple["Qubit", "Qubit", "Qubit"]:
        """Used primarily in __setitem__ to 'pull out' the Qubit to be replaced, returns three Qubit objects that can be recombined"""
        if qubit_index is not None:
            if qubit_index > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
            if isinstance(qubit_index, int):
                A_rho = self.partial_trace(0, self.n - qubit_index)
                B_rho = self.partial_trace(qubit_index + 1,0)
                isolated_rho = self.partial_trace(qubit_index, self.n - qubit_index - 1)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit_index)}, expected int") 
            return A_rho, isolated_rho, B_rho
        isolated_rho = self.partial_trace(qubit_index, self.n - qubit_index - 1)
    
    def norm(self) -> None:
        """Normalises a rho matrix, returns type None"""
        if self.rho.shape[0] == self.rho.shape[1]:
            trace_rho = np.trace(self.rho)
            if trace_rho != 0:
                self.rho = self.rho / trace_rho
            else:
                raise QuantumStateError(f"The trace of the density matrix cannot be 0 and so cannot normalise")
        raise QuantumStateError(f"self.rho must be a square matrix, not of shape {self.rho.shape}")

    def set_display_mode(self, mode: str) -> None:
        """Sets the display mode between the three options, returns type None"""
        if mode not in ["vector", "density", "both"]:
            raise QuantumStateError(f"The display mode must be set in 'vector', 'density' or 'both'")
        self.display_mode = mode

    def build_pure_rho(self) -> np.ndarray:
        """Builds a pure rho matrix, primarily in initiation of Qubit object, returns type np.ndarray"""
        if isinstance(self.state, np.ndarray):
            return np.einsum("i,j", np.conj(self.state), self.state, optimize=True)
        raise StatePreparationError(f"self.state cannot be of type {type(self.state)}, expected np.ndarray")
    
    def build_mixed_rho(self) -> np.ndarray:
        """Builds a mixed rho matrix, primarily in initiation of Qubit object, returns type np.ndarray"""
        if self.weights is not None:
            mixed_rho = np.zeros((len(self.state[0]),len(self.state[0])), dtype=np.complex128)
            for i in range(len(self.weights)):
                mixed_rho += self.weights[i] * np.einsum("j,k->jk", np.conj(self.state[i]), self.state[i], optimize=True)
            return mixed_rho
        raise StatePreparationError(f"For a mixed rho to be made, you must provide weights in kwargs")
        
    def build_state_from_rho(self) -> np.ndarray:
        """Builds a state vector from the given rho matrix, primarily for printing purposes, returns type np.ndarray"""
        if sparse.issparse(self.rho):
            N = self.rho.shape[0]
            k = max(1, N - 2)
            probs, states = eigsh(self.rho, k=k, which="LM")
            probs = sparse.csr_matrix(probs, dtype=np.float64)
            states = sparse.csr_matrix(states, dtype=np.complex128)
        else:
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
    
    def debug(self, title=True) -> None:
        """Prints out lots of information on the Qubits core properties primarily for debug purposes, returns type None"""
        print(f"\n")
        if title:
            print("-" * linewid)
            print(f"QUBIT DEBUG")
        print(f"self.rho.shape: {self.rho.shape}")
        print(f"self.rho type: {type(self.rho)}")
        print(f"self.rho:\n {self.rho}")
        print(f"self.state:\n {self.build_state_from_rho()}")
        print(f"self.n: {self.n}")
        for i in range(self.n):
            print(f"Qubit {i}: {self[i]}")
        print(f"state_type: {self.state_type}")
        print(f"All attributes and variables of the Qubit object:")
        print(vars(self))
        if title:
            print("-" * linewid)

    
            


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

