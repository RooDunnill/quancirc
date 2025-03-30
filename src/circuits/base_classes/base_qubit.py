from .base_class_utilities.base_class_errors import BaseQuantumStateError, BaseStatePreparationError
from ..circuit_config import *
from ..circuit_utilities.sparse_funcs import *
import sympy as sp

__all__ = ["BaseQubit"]

class BaseQubit:
    def __init__(self, **kwargs):
        self.skip_val = kwargs.get("skip_validation", False)
        self.display_mode = kwargs.get("display_mode", "density")
        self.name: str = kwargs.get("name","|Quantum State>")
        self.state = kwargs.get("state", None)

    def __str__(self: "BaseQubit") -> str:
        state_print = self.build_state_from_rho() if self.class_type == "qubit" else self.state
        rho = dense_mat(self.rho) if self.class_type == "qubit" else self.build_pure_rho()
        rho_str = np.array2string(rho, precision=p_prec, separator=', ', suppress_small=True)
        if not self.name:
            self.name = f"{self.state_type} {self.class_type}"
        if self.state_type == "pure":
            if isinstance(state_print, tuple):
                raise BaseStatePreparationError(f"The state vector of a pure state cannot be a tuple")
            state_str = np.array2string(dense_mat(state_print), precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"{self.name}:\n{state_str}"
            elif self.display_mode == "density":
                return f"{self.name}\n{rho_str}" 
            elif self.display_mode == "both":
                return f"{self.name}\nState:\n{state_str}\nRho:\n{rho_str}"
        elif self.state_type == "mixed":
            if isinstance(state_print, np.ndarray):
                raise BaseStatePreparationError(f"The state vector of a mixed state cannot be a sinlge np.ndarray")
            weights = dense_mat(state_print[0])
            state = dense_mat(state_print[1])
            weights_str = np.array2string(weights, precision=p_prec, separator=', ', suppress_small=True)
            state_str = np.array2string(state, precision=p_prec, separator=', ', suppress_small=True)
            if self.display_mode == "vector":
                return f"{self.name}\nWeights\n{weights_str}\nStates:\n{state_str}"
            elif self.display_mode == "density":
                return  f"{self.name}\nRho:\n{rho_str}"
            elif self.display_mode == "both":
                return f"{self.name}\nWeights\n{weights_str}\nStates:\n{state_str}\nRho:\n{rho_str}"
        elif self.state_type == "non unitary":
            return f"Non Quantum State Density Matrix:\n{rho_str}"
        
    def build_pure_rho(self):
        """Builds a pure rho matrix, primarily in initiation of Qubit object, returns type sp.MatrixBase"""
        if isinstance(self.state, sp.MatrixBase):
            return self.state * self.state.H
        elif isinstance(self.state, np.ndarray):
            return np.einsum("i,j", np.conj(self.state), self.state, optimize=True)
        
    def __repr__(self: "BaseQubit") -> str:
        return self.__str__()

    def __copy__(self: "BaseQubit") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem")
    
    def __deepcopy__(self: "BaseQubit") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem, its twice the sin to try to double copy them")
    
    def __isub__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            self = self - other
            return self
        raise BaseQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __iadd__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            self = self + other
            return self
        raise BaseQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def set_display_mode(self: "BaseQubit", mode: str) -> None:
        """Sets the display mode between the three options, returns type None"""
        if mode not in ["vector", "density", "both"]:
            raise BaseQuantumStateError(f"The display mode must be set in 'vector', 'density' or 'both'")
        self.display_mode = mode

    def __imod__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            self = self % other
            return self
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected types Qubit and Qubit")

    def debug(self: "BaseQubit", title=True) -> None:
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