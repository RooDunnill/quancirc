from .base_class_utilities.base_class_errors import BaseQuantumStateError
from ..circuit_config import *



class BaseQubit:
    def __copy__(self: "BaseQubit") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem")
    
    def __deepcopy__(self: "BaseQubit") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem, its twice the sin to try to double copy them")
    
    def __isub__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            self.rho = self.rho - other.rho
            return self
        raise BaseQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __iadd__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            self.rho = self.rho + other.rho
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