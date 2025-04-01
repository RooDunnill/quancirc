from .base_class_utilities.base_class_errors import BaseQuantumStateError, BaseStatePreparationError
from ..circuit_config import *
from ..circuit_utilities.sparse_funcs import *
import sympy as sp

__all__ = ["BaseQubit"]


def combine_qubit_attr(self: "BaseQubit", other: "BaseQubit", op: str = None) -> dict:
        """Allows the returned objects to still return name and info too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            if op == "%":
                self_name_size = int(np.log2(self.dim))
                other_name_size = int(np.log2(other.dim))
                new_name = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>"
                kwargs["name"] = new_name
            elif op == "@":
                new_name = f"{self.name} {other.name}"
                kwargs["name"] = new_name
            elif op:
                new_name = f"{self.name} {op} {other.name}"
            else:
                new_name = f"{self.name}"
            if len(new_name) > name_limit:
                new_name = new_name[len(new_name) - name_limit:]
            kwargs["name"] = new_name
        if isinstance(self, BaseQubit) and isinstance(other, BaseQubit):
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
        elif isinstance(other, BaseQubit):
            if hasattr(other, "index"):
                kwargs["index"] = other.index
        if hasattr(self, "skip_val") and self.skip_val == True:
            kwargs["skip_validation"] = True
        elif hasattr(other, "skip_val") and other.skip_val == True: 
            kwargs["skip_validation"] = True
        return kwargs

def copy_qubit_attr(self: "BaseQubit") -> dict:
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




class BaseQubit:
    all_immutable_attr = ["class_type"]
    def __init__(self, **kwargs):
        self.skip_val = kwargs.get("skip_validation", False)
        self.display_mode = kwargs.get("display_mode", "density")
        self.name: str = kwargs.get("name","|\u03C8>")
        self.state = kwargs.get("state", None)
        self.index = None

    def __str__(self: "BaseQubit") -> str:
        rho = dense_mat(self.rho) if self.class_type == "qubit" else self.build_pure_rho()
        rho_str = np.array2string(rho, precision=p_prec, separator=', ', suppress_small=True)
        if not self.name:
            self.name = f"{self.state_type} {self.class_type}"
        if self.display_mode == "density":
            return f"{self.state_type} {self.name}\n{rho_str}" 
        state_print = self.build_state_from_rho() if self.class_type == "qubit" else self.state
        if isinstance(state_print, tuple):
            raise BaseStatePreparationError(f"The state vector of a pure state cannot be a tuple")
        state_str = np.array2string(dense_mat(state_print), precision=p_prec, separator=', ', suppress_small=True)
        if self.display_mode == "vector":
            return f"{self.name}:\n{state_str}"
        elif self.display_mode == "both":
            return f"{self.name}\nState:\n{state_str}\nRho:\n{rho_str}"
        
    def __setattr__(self: "BaseQubit", name: str, value) -> None:
        if name == "immutable":
            object.__setattr__(self, name, True)
        if not hasattr(self, "_initialised"): 
            object.__setattr__(self, name, value)
            return
        if hasattr(self, "immutable") and name in self.immutable_attr:
            current_value = getattr(self, name, None)
            if name == "rho" or name == "state":
                print(f"Dealing with attribute {name}")
                if np.array_equal(dense_mat(current_value), dense_mat(value)):
                    object.__setattr__(self, name, value)
                    return
            elif current_value == value:
                return
            raise AttributeError(f"Cannot modify immutable object: {name}")
        if name in self.all_immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        object.__setattr__(self, name, value)
     
        
    def set_state_type(self) -> None:
        """Checks that state type and corrects if needed, returns type None"""
        excluded_classes = ["lwqubit", "symbqubit"]
        if self.class_type in excluded_classes:
            return
        purity = (self.rho.dot(self.rho)).diagonal().sum().real if sparse.issparse(self.rho) else np.einsum('ij,ji', self.rho, self.rho).real  
        if self.skip_val:
            self.state_type = "Non-Unitary"
            self.set_display_mode("density")
        elif np.isclose(purity, 1.0, atol=1e-4):
            self.state_type = "Pure"
        elif purity < 1:
            self.state_type = "Mixed"
            self.set_display_mode("density")
        else:
            raise BaseStatePreparationError(f"The purity of a state must be between 0 and 1, purity: {purity}")
        
    def build_pure_rho(self):
        """Builds a pure rho matrix, primarily in initiation of Qubit object, returns type sp.MatrixBase"""
        if isinstance(self.state, sp.MatrixBase):
            return self.state * self.state.H
        elif isinstance(self.state, np.ndarray):
            return np.einsum("i,j", np.conj(self.state), self.state, optimize=True)
        
    def set_display_mode(self: "BaseQubit", mode: str) -> None:
        """Sets the display mode between the three options, returns type None"""
        if mode not in ["vector", "density", "both"]:
            raise BaseQuantumStateError(f"The display mode must be set in 'vector', 'density' or 'both'")
        non_vector_state_types = ["Mixed", "Non-Unitary"]
        if self.state_type in non_vector_state_types and mode != "density":
            raise BaseQuantumStateError(f"Mixed and Non-Unitary states can only be displayed as density matrices")
        self.display_mode = mode
        
    def __repr__(self: "BaseQubit") -> str:
        return self.__str__()

    def __copy__(self: "BaseQubit") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem")
    
    def __deepcopy__(self: "BaseQubit") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem, its twice the sin to try to double copy them")
    
    def __mul__(self: "BaseQubit", other: int | float) -> "BaseQubit":
        if isinstance(other, (int, float)):
            new_rho = self.rho * other
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "*"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __rmul__(self: "BaseQubit", other: int | float) -> "BaseQubit":
        return self.__mul__(other)
    
    def __imul__(self: "BaseQubit", other: float) -> "BaseQubit":
        if isinstance(other, (int, float)):
            self.rho *= other
            return self
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
    def __truediv__(self: "BaseQubit", other: int | float) -> "BaseQubit":
        if isinstance(other, (int, float)):
            new_rho = self.rho / other
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "*"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __itruediv__(self: "BaseQubit", other: float) -> "BaseQubit":
        if isinstance(other, (int, float)):
            self.rho /= other
            return self
        raise QuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
    def __sub__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        """Subtraction of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, BaseQubit):
            new_rho = self.rho - other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}                #CAREFUL skip val here
            kwargs.update(combine_qubit_attr(self, other, op = "-"))
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __isub__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            if hasattr(self, "immutable"):
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self - other
            return self
        raise BaseQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __add__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        """Addition of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, BaseQubit):
            new_rho = self.rho + other.rho 
            kwargs = {"rho": new_rho, "skip_validation": True}
            kwargs.update(combine_qubit_attr(self, other, op = "+"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __iadd__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            if hasattr(self, "immutable"):
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self + other
            return self
        raise BaseQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")

    def __imod__(self: "BaseQubit", other: "BaseQubit") -> "BaseQubit":
        if isinstance(other, BaseQubit):
            if hasattr(self, "immutable"):
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self % other
            return self
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected types Qubit and Qubit")

    def debug(self: "BaseQubit", title=True) -> None:
        """Prints out lots of information on the Qubits core properties primarily for debug purposes, returns type None"""
        print(f"\n")
        if title:
            print("-" * linewid)
            print(f"QUBIT DEBUG")
        if self.rho is not None:
            print(f"self.rho.shape: {self.rho.shape}")
            print(f"self.rho type: {type(self.rho)}")
            print(f"self.rho:\n {self.rho}")
            print(f"self.n: {self.n}")
            for i in range(self.n):
                print(f"Qubit {i}: {self[i]}")
        print(f"self.state:\n {self.build_state_from_rho()}") if self.rho is not None else print(f"self.state:\n {self.state}")
        print(f"state_type: {self.state_type}")
        print(f"All attributes and variables of the Qubit object:")
        print(vars(self))
        if title:
            print("-" * linewid)