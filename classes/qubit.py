import numpy as np
from utilities import QuantumStateError, StatePreparationError, linewid
from .static_methods.qubit_methods import *
from utilities.config import p_prec


def combine_qubit_attr(self, other, op: str = None):
        """Allows the returned objects to still return name and info too"""
        kwargs = {}
        if op == "@":
            self_name_size = int(np.log2(self.dim))
            other_name_size = int(np.log2(other.dim)) 
            kwargs["name"] = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>"
        elif hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            if op:
                kwargs["name"] = f"{self.name} {op} {other.name}"
            else:
                kwargs["name"] = f"{self.name}"
        if hasattr(self, "state_type") and hasattr(other, "state_type"):
            if self.state_type == "mixed" or other.state_type == "mixed":
                kwargs["state_type"] = "mixed"
            elif self.state_type == "pure" or other.state_type == "pure":
                kwargs["state_type"] = "pure"
            else:
                raise QuantumStateError(f"The state types must be either 'pure' or 'mixed', not {self.state_type} and {other.state_type}")
        return kwargs

def copy_qubit_attr(qubit, **overrides):
    attributes = vars(qubit).copy()
    attributes.update(overrides)
    return qubit.__class__(**attributes)

class Qubit:                                           #creates the qubit class
    """The class to define and initialise Qubits and Quantum States"""

    def __init__(self, **kwargs) -> None:
        self.class_type = "qubit"
        self.name: str = kwargs.get("name", None)
        self.state_type: str = kwargs.get("type", "pure")                   #the default qubit is a single pure qubit |0>
        self.display_mode = kwargs.get("display_mode", "vector")
        self.weights: list = kwargs.get("weights", None)
        self.name: str = kwargs.get("name","|Quantum State>")
        self.state: list = kwargs.get("state", None)
        self.rho: list = kwargs.get("rho", None)

        if self.state is not None:
            if not isinstance(self.state, (list, np.ndarray)):
                raise StatePreparationError(f"The inputted self.state cannot be of type {type(self.state)}, expected list or np.ndarray")
            self.state = np.array(self.state, dtype=np.complex128)

        if self.state is not None and self.rho is None:
            if self.state_type == "pure":
                self.rho = self.build_pure_rho()
            elif self.state_type == "mixed":
                self.rho = self.build_mixed_rho()

        if self.rho is not None:
            if not isinstance(self.rho, (list, np.ndarray)):
                raise StatePreparationError(f"The inputted self.rho cannot be of type {type(self.rho)}, expected list or np.ndarray")
            self.rho = np.array(self.rho, dtype=np.complex128)
        else:
            raise StatePreparationError(f"The initialised object must have atleast 1 of the following: a state vector or a density matrix")

        
        self.dim = len(self.rho)
        self.length = self.dim ** 2
        self.n = int(np.log2(self.dim))
        self.skip_val = kwargs.get("skip_validation", False)
        if not self.skip_val:
            self.is_valid_density_matrix()

    def __str__(self):
        if self.skip_val:
            self.display_mode = "density"
        rho_str = np.array2string(self.rho, precision=p_prec, separator=', ', suppress_small=True)
        if self.state_type == "pure":
            state_str = np.array2string(self.build_state_from_rho(), precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"Pure Quantum State Vector:\n{state_str}"
            elif self.display_mode == "density":
                return f"Pure Quantum State Density Matrix:\n{rho_str}"
            elif self.display_mode == "both":
                return f"Pure Quantum State Vector:\n{state_str}\n\nDensity Matrix:\n{rho_str}"

        elif self.state_type == "mixed":
            weights, state = self.build_state_from_rho()
            weights_str = np.array2string(weights, precision=p_prec, separator=', ', suppress_small=True)
            state_str = np.array2string(state, precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"Mixed Quantum State Vector:\nWeights:\n{weights_str}\n\nStates:\n{state_str}"
            elif self.display_mode == "density":
                return f"Mixed Quantum State Density Matrix:\n{rho_str}"
            elif self.display_mode == "both":
                return f"Mixed Quantum State Vector:\nWeights:\n{weights_str}\n\nStates:\n{state_str}\n\nDensity Matrix:\n{rho_str}"
    
    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            new_rho = np.kron(self.rho, other.rho)
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
        
    def __mul__(self, other):      #gateT @ rho @ gate
        if isinstance(other, self.__class__):
            raise QuantumStateError(f"Cannot matrix multiply (double) two Quantum states together")
        elif other.class_type == "gate":
            new_rho = np.dot(np.dot(np.conj(other.matrix), self.rho), other.matrix)
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "@"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")
    
    def __or__(self, other):       #rho @ gate
        if isinstance(other, self.__class__):
            raise QuantumStateError(f"Cannot matrix multiply (singular) two Quantum states together")
        elif other.class_type == "gate":
            new_rho = np.dot(self.rho, other.matrix)
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "|"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Gate, Qubit or np.ndarray")

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            new_rho = self.rho - other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "-"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_rho = self.rho + other.rho
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "+"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __and__(self, other):
        if isinstance(other, self.__class__):
            new_rho = np.block([[self.rho, np.zeros_like(other.rho)], [np.zeros_like(self.rho), other.rho]])
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "&"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.rho is not None and other.rho is not None:
                return np.allclose(self.rho, other.rho)
            raise QuantumStateError(f"The inputted objects must have attr: self.rho and other.rho")
        raise QuantumStateError(f"Cannot have types {type(self)} and {type(other)}, expected two Qubit classes")
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(self.n)]
        elif isinstance(index, int):
            get_qubit = self.isolate_qubit(index)
            if get_qubit is None:
                raise QuantumStateError(f"Could not isolate qubit {index}, invalid index input")
            return get_qubit
        raise QuantumStateError(f"Index cannot be of type {type(index)}, expected type int or slice")
    
    def __setitem__(self, index, new_rho):
        if not isinstance(self.rho, np.ndarray):
            raise QuantumStateError(f"self.rho cannot be of type {type(self.rho)}, must be of type np.ndarray")
        rho_A, replaced_qubit, rho_B = self.decompose_state(index)
        if replaced_qubit.dim == new_rho.dim:
            new_state = rho_A @ new_rho @ rho_B 
            self.rho = new_state.rho
        else:
            raise QuantumStateError(f"The dimensions of the new qubit must be the same as the dimensions of the old qubit")

    def partial_trace(self, **kwargs) -> np.ndarray:
        """Computes the partial trace of a state, can apply a trace from either 'side' and can trace out an arbitrary amount of qubits
        Args:
            self: The density instance
            **kwargs
            trace_out_system:str : Chooses between A and B which to trace out, defaults to B
            state_size:int : Chooses the number of Qubits in the trace out state, defaults to 1 Qubit
        Returns:self.rho_a if trace_out_system = B
                self.rho_b if trace_out_system = A"""
        trace_out_system = kwargs.get("trace_out", None)
        trace_out_state_size = kwargs.get("state_size", None)
        rho = kwargs.get("rho", self.rho)
        rho_dim = len(rho)
        rho_n = int(np.log2(rho_dim))
        if trace_out_state_size == rho_n:
            return Qubit(rho=np.array([1]))
        if trace_out_state_size is not None:
            trace_out_state_size = int(trace_out_state_size)
        
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        new_mat = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        traced_out_dim_range = np.arange(traced_out_dim)
        if isinstance(rho, np.ndarray):
            kwargs = vars(self).copy()
            if trace_out_system == "B":
                    for k in range(reduced_dim):
                        for i in range(reduced_dim):           #the shapes of tracing A and B look quite different but follow a diagonalesc pattern
                            new_mat[i, k] = np.sum(rho[traced_out_dim_range+i*traced_out_dim, traced_out_dim_range+k*traced_out_dim])
                    return copy_qubit_attr(self, rho=new_mat)
            elif trace_out_system == "A":
                    for k in range(reduced_dim):
                        for i in range(reduced_dim):
                            new_mat[i, k] = np.sum(rho[reduced_dim*traced_out_dim_range+i, reduced_dim *traced_out_dim_range+k])
                    return copy_qubit_attr(self, rho=new_mat)

    def isolate_qubit(self, qubit):
        if qubit is not None:
            if qubit > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit}, must be no more than the number of qubits in the state: {self.n}")
            if qubit == 0:
                isolated_rho = self.partial_trace(trace_out="B", state_size = self.n - 1)
            elif qubit == self.n - 1:
                isolated_rho = self.partial_trace(trace_out="A", state_size = self.n - 1)
            elif isinstance(qubit, int):
                A_rho = self.partial_trace(trace_out="B", state_size = self.n - qubit - 1)
                A_n = int(np.log2(len(A_rho.rho)))
                isolated_rho = self.partial_trace(rho=A_rho.rho, trace_out="A", state_size = A_n - 1)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit)}, expected int") 
            return isolated_rho
        
    def decompose_state(self, qubit):
        if qubit is not None:
            if qubit > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit}, must be no more than the number of qubits in the state: {self.n}")
            if isinstance(qubit, int):
                temp_rho = self.partial_trace(trace_out="B", state_size = self.n - qubit - 1)
                A_rho = self.partial_trace(trace_out="B", state_size = self.n - qubit)
                B_rho = self.partial_trace(trace_out="A", state_size = qubit + 1)
                temp_n = int(np.log2(len(temp_rho.rho)))
                isolated_rho = self.partial_trace(rho=temp_rho.rho, trace_out="A", state_size = temp_n - 1)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit)}, expected int") 
            return A_rho, isolated_rho, B_rho
    
    def norm(self):
        if self.rho.shape[0] == self.rho.shape[1]:
            trace_rho = np.trace(self.rho)
            if trace_rho != 0:
                self.rho = self.rho / trace_rho
            else:
                raise QuantumStateError(f"The trace of the density matrix cannot be 0 and so cannot normalise")
        raise QuantumStateError(f"self.rho must be a square matrix, not of shape {self.rho.shape}")

    def set_display_mode(self, mode):
        if mode not in ["vector", "density", "both"]:
            raise QuantumStateError(f"The display mode must be set in 'vector', 'density' or 'both'")
        self.display_mode = mode

    def build_pure_rho(self):
        if isinstance(self.state, np.ndarray):
            return np.outer(np.conj(self.state), self.state)
        raise QuantumStateError(f"self.state cannot be of type {type(self.state)}, expected np.ndarray")
    
    def build_mixed_rho(self):
        if self.weights is not None:
            mixed_rho = np.zeroes(len(self.state[0])**2, dtype=np.complex128)
            for i in range(len(self.weights)):
                mixed_rho += self.weights[i] * np.outer(np.conj(self.state[i]), self.state[i])
            return mixed_rho
        raise QuantumStateError(f"For a mixed rho to be made, you must provide weights in kwargs")
        
    def build_state_from_rho(self):
        probs, states = np.linalg.eigh(self.rho)
        max_eigenvalue = np.argmax(np.isclose(probs, 1.0))
        if np.any(probs) > 1.0:
            raise QuantumStateError(f"You cannot have a probability over 1, the probabilities {probs} have been computed incorrectly")
        if np.any(np.isclose(probs, 1.0)):
            state_vector = states[:, max_eigenvalue]
            norm = np.linalg.norm(state_vector)
            if norm != 0:
                state_vector /= norm
            return state_vector
        self.state_type = "mixed"
        return probs, states
    
    def debug(self):
        print(f"\n")
        print("-" * linewid)
        print(f"QUBIT CLASS DEBUG")
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
        print("-" * linewid)

    def is_valid_density_matrix(self):
        if not np.allclose(self.rho, self.rho.conj().T):  
            raise QuantumStateError(f"Density matrix is not Hermitian: {self.rho}")
        if not np.array_equal(self.rho, np.array([1])):
            eigenvalues = np.linalg.eigvalsh(self.rho)
            if np.any(eigenvalues < -1e-4):
                negative_indices = np.where(eigenvalues < 0)[0]
                raise QuantumStateError(f"Density matrix is not positive semi-definite. "
                                     f"Negative eigenvalues found at indices {negative_indices}")
            if not np.isclose(np.trace(self.rho), 1.0):
                raise QuantumStateError(f"Density matrix must have a trace of 1, not of trace {np.trace(self.rho)}")

    

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
q1 = Qubit.q1()
qp = Qubit.qp()
qm = Qubit.qm()
qpi = Qubit.qpi()
qmi = Qubit.qmi()

