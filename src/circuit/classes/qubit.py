import numpy as np
from ..circuit_utilities.circuit_errors import QuantumStateError, StatePreparationError
from .static_methods.qubit_methods import *
from ..circuit_config import *
from ..circuit_utilities.validation_funcs import qubit_validation, rho_validation


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
    def __init__(self, **kwargs) -> None:
        object.__setattr__(self, 'class_type', 'qubit')
        self.skip_val = kwargs.get("skip_validation", False)
        self.display_mode = kwargs.get("display_mode", "vector")
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

        self.dim = len(self.rho)
        self.length = self.dim ** 2
        self.n = int(np.log2(self.dim))

    def set_state_type(self) -> None:
        """Checks that state type and corrects if needed, returns type None"""
        purity = np.trace(np.dot(self.rho, self.rho)).real
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
            if self.weights:
                self.rho = self.build_mixed_rho()
            else:
                self.rho = self.build_pure_rho()

    def __repr__(self):
        return self.__str__()


    def __str__(self) -> str:
        state_print = self.build_state_from_rho()
        rho_str = np.array2string(self.rho, precision=p_prec, separator=', ', suppress_small=True)
        if self.state_type == "pure":
            if isinstance(state_print, tuple):
                raise StatePreparationError(f"The state vector of a pure state cannot be a tuple")
            state_str = np.array2string(state_print, precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"Pure Quantum State Vector:\n{state_str}"
            elif self.display_mode == "density":
                return f"Pure Quantum State Density Matrix:\n{rho_str}"
            elif self.display_mode == "both":
                return f"Pure Quantum State Vector:\n{state_str}\n\nDensity Matrix:\n{rho_str}"

        elif self.state_type == "mixed":
            if isinstance(state_print, np.ndarray):
                raise StatePreparationError(f"The state vector of a mixed state cannot be a sinlge np.ndarray")
            weights = state_print[0]
            state = state_print[1]
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
            new_rho = np.kron(self.rho, other.rho)
            new_rho = np.round(new_rho, decimals=10)
            kwargs = {"rho": new_rho}
            kwargs.update(combine_qubit_attr(self, other, op = "%"))
            return Qubit(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
        
    def __matmul__(self: "Qubit", other: "Qubit") -> "Qubit":      #gateT @ rho @ gate
        """Matrix multiplication between two Qubit objects, returns a Qubit object"""
        if isinstance(other, Qubit):
            raise QuantumStateError(f"Cannot matrix multiply (double) two Quantum states together")
        elif other.class_type == "gate":
            new_rho = np.dot(np.dot(other.matrix, self.rho), np.conj(other.matrix.T))
            new_rho = np.round(new_rho, decimals=10)
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
            return get_qubit
        raise QuantumStateError(f"Index cannot be of type {type(index)}, expected type int or slice")
    
    def __setitem__(self: "Qubit", index: int, new_state: "Qubit") -> None:
        """Sets a single Qubit to the inputted Qubit, then tensors the state back into the multistate, returns None type"""
        if not isinstance(self.rho, np.ndarray):
            raise QuantumStateError(f"self.rho cannot be of type {type(self.rho)}, must be of type np.ndarray")
        rho_A, replaced_qubit, rho_B = self.decompose_state(index)
        if replaced_qubit.dim == new_state.dim:
            new_state = rho_A % new_state % rho_B 
            if new_state.dim == self.dim:
                self.rho = new_state.rho

            else:
                raise QuantumStateError(f"The new rho needs to be the same dimension as the old rho, not {self.dim} and {new_state.dim}")
        else:
            raise QuantumStateError(f"The dimensions of the new qubit must be the same as the dimensions of the old qubit")

    def partial_trace(self, trace_out_system: str, trace_out_state_size: int, **kwargs) -> "Qubit":
        """Computes the partial trace of a state, can apply a trace from either 'side' and can trace out an arbitrary amount of qubits
        Args:
            self: The density instance
            **kwargs
            trace_out_system:str : Chooses between A and B which to trace out, defaults to B
            state_size:int : Chooses the number of Qubits in the trace out state, defaults to 1 Qubit
        Returns:self.rho_a if trace_out_system = B
                self.rho_b if trace_out_system = A"""
        rho = kwargs.get("rho", self.rho)
        if not isinstance(rho, (np.ndarray, list)):
            raise QuantumStateError(f"rho cannot be of type {type(rho)}, expected type np.ndarray or type list")
        if trace_out_system not in ["A", "B"]:
            raise QuantumStateError(f"trace_out_system must be either str: 'A' or 'B', cannot be {trace_out_system}")
        if not isinstance(trace_out_state_size, int):
            raise QuantumStateError(f"trace_out_state_size cannot be of type {type(trace_out_state_size)}, expected type int")
        rho_dim = len(rho)
        rho_n = int(np.log2(rho_dim))
        if trace_out_state_size == rho_n:
            return Qubit()
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        new_rho = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        rho_dim = len(rho)
        rho_n = int(np.log2(rho_dim))
        if trace_out_state_size == rho_n:
            kwargs.update(copy_qubit_attr(self))
            return Qubit(**kwargs)
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        new_rho = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        traced_out_dim_range = np.arange(traced_out_dim)
        if trace_out_system == "B":
                for k in range(reduced_dim):
                    for i in range(reduced_dim):           #the shapes of tracing A and B look quite different but follow a diagonalesc pattern
                        new_rho[i, k] = np.sum(rho[traced_out_dim_range+i*traced_out_dim, traced_out_dim_range+k*traced_out_dim])
        elif trace_out_system == "A":
            for k in range(reduced_dim):
                for i in range(reduced_dim):
                    new_rho[i, k] = np.sum(rho[reduced_dim*traced_out_dim_range+i, reduced_dim *traced_out_dim_range+k])
        kwargs = {"rho": new_rho}
        kwargs.update(copy_qubit_attr(self))
        return Qubit(**kwargs)
    
    def partial_trace_gen(self, size_a, size_c, **kwargs):
        rho = kwargs.get("rho", self.rho)
        if not isinstance(rho, (np.ndarray, list)):
            raise QuantumStateError(f"rho cannot be of type {type(rho)}, expected type np.ndarray or type list")
        dim_a = int(2**size_a)
        dim_c = int(2**size_c)
        rho_dim = len(rho)
        dim_b = int(rho_dim/(dim_a*dim_c))
        if size_c == 0:
            rho_reshape = rho.reshape(dim_a, dim_b, dim_a, dim_b)
            print(rho_reshape.shape)
            new_rho = np.einsum("abcd->bd", rho_reshape)
        elif size_a == 0:
            rho_reshape = rho.reshape(dim_b, dim_c, dim_b, dim_c)
            print(rho_reshape.shape)
            new_rho = np.einsum("abcd->ac", rho_reshape)
        else:
            rho_reshape = rho.reshape(dim_a, dim_b, dim_c, dim_a, dim_b, dim_c)
            new_rho = np.einsum("abcdef->be", rho_reshape)
        kwargs = {"rho": new_rho}
        kwargs.update(copy_qubit_attr(self))
        return Qubit(**kwargs)


    def isolate_qubit(self, qubit_index: int) -> "Qubit":
        """Used primarily in __getitem__ to return a single Qubit from a multiqubit state, returns a Qubit object"""
        if qubit_index is not None and isinstance(qubit_index, int):
            if qubit_index > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
            if qubit_index == 0:
                isolated_rho = self.partial_trace("B", self.n - 1)
            elif qubit_index == self.n - 1:
                isolated_rho = self.partial_trace("A", self.n - 1)
            elif isinstance(qubit_index, int):
                A_rho = self.partial_trace("B", self.n - qubit_index - 1)
                A_n = int(np.log2(len(A_rho.rho)))
                isolated_rho = self.partial_trace("A", A_n - 1, rho=A_rho.rho)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit_index)}, expected int") 
            return isolated_rho
        
    def decompose_state(self, qubit_index: int) -> tuple["Qubit", "Qubit", "Qubit"]:
        """Used primarily in __setitem__ to 'pull out' the Qubit to be replaced, returns three Qubit objects that can be recombined"""
        if qubit_index is not None:
            if qubit_index > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
            if isinstance(qubit_index, int):
                temp_rho = self.partial_trace("B", self.n - qubit_index - 1)
                A_rho = self.partial_trace("B", self.n - qubit_index)
                B_rho = self.partial_trace("A", qubit_index + 1)
                temp_n = int(np.log2(len(temp_rho.rho)))
                isolated_rho = self.partial_trace("A", temp_n - 1, rho=temp_rho.rho)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit_index)}, expected int") 
            return A_rho, isolated_rho, B_rho
    
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
            return np.outer(np.conj(self.state), self.state)
        raise StatePreparationError(f"self.state cannot be of type {type(self.state)}, expected np.ndarray")
    
    def build_mixed_rho(self) -> np.ndarray:
        """Builds a mixed rho matrix, primarily in initiation of Qubit object, returns type np.ndarray"""
        if self.weights is not None:
            mixed_rho = np.zeros(len(self.state[0])**2, dtype=np.complex128)
            for i in range(len(self.weights)):
                mixed_rho += self.weights[i] * np.outer(np.conj(self.state[i]), self.state[i])
            return mixed_rho
        raise StatePreparationError(f"For a mixed rho to be made, you must provide weights in kwargs")
        
    def build_state_from_rho(self) -> np.ndarray:
        """Builds a state vector from the given rho matrix, primarily for printing purposes, returns type np.ndarray"""
        probs, states = np.linalg.eigh(self.rho)
        probs = np.array(probs, dtype=np.float64)
        states = np.array(states, dtype=np.complex128)
        max_eigenvalue = np.argmax(np.isclose(probs, 1.0))
        if np.any(probs) > 1.0:
            raise QuantumStateError(f"You cannot have a probability over 1, the probabilities {probs} have been computed incorrectly")
        if np.any(np.isclose(probs, 1.0)):
            state_vector = states[:, max_eigenvalue]
            norm = np.linalg.norm(state_vector)
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

