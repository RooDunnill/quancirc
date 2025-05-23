import logging
import sympy as sp
import copy
from abc import ABC
from .base_class_utilities.base_class_errors import BaseQuantumStateError, BaseStatePreparationError
from ..circuit_config import *
from ..circuit_utilities.sparse_funcs import *
from .base_class_utilities.validation_funcs import base_quant_state_validation

__all__ = ["BaseQuantState"]


def combine_quant_state_attr(self: "BaseQuantState", other: "BaseQuantState", kwargs: dict=None) -> dict:
        """Allows the returned objects to still return name and info too"""
        kwargs = {} if kwargs==None else kwargs
        if isinstance(self, BaseQuantState) and isinstance(other, BaseQuantState):
            if hasattr(self, "display_mode") and hasattr(other, "display_mode"):
                if self.display_mode == "both" or other.display_mode == "both":
                    kwargs["display_mode"] = "both"
                elif self.display_mode == "density" or other.display_mode == "density":
                    kwargs["display_mode"] = "density"
                else:
                    kwargs["display_mode"] = "vector"

        if hasattr(self, "skip_val") and self.skip_val == True:
            kwargs["skip_val"] = True
        elif hasattr(other, "skip_val") and other.skip_val == True:
            kwargs["skip_val"] = True
        if hasattr(self, "id") and self.id and isinstance(self.id, int):
            if hasattr(self, "history"):
                kwargs["history"] = copy.deepcopy(self.history)
            kwargs["id"] = copy.deepcopy(self.id)
        if self.class_type=="gate" and hasattr(other, "id") and isinstance(other.id, int):
            if hasattr(other, "history"):
                kwargs["history"] = copy.deepcopy(other.history)
            kwargs["id"] = copy.deepcopy(other.id)
        logging.debug(f"Carrying over kwargs: {kwargs}")
        return kwargs

def copy_quant_state_attr(self: "BaseQuantState", kwargs: dict=None) -> dict:
    kwargs = {} if kwargs==None else kwargs
    if "id" not in kwargs and hasattr(self, "id") and isinstance(self.id, int):
        kwargs["id"] = copy.deepcopy(self.id)
        if "history" not in kwargs and hasattr(self, "history"):
            kwargs["history"] = copy.deepcopy(self.history)
    if "display_mode" not in kwargs and hasattr(self, "display_mode"):
        kwargs["display_mode"] = copy.deepcopy(self.display_mode)
    if "skip_val" not in kwargs and hasattr(self, "skip_val") and self.skip_val == True:
        kwargs["skip_val"] = True   #avoids deepcopy
    logging.debug(f"Carrying over kwargs: {kwargs}")
    return kwargs


@log_all_methods
class BaseQuantState(ABC):
    qubit_counter = 0
    all_immutable_attr = ["class_type"]

    def __init__(self, **kwargs):
        logging.debug(f"Creating a BaseQuantState instance with kwargs {kwargs}")
        self.id = kwargs.get("id", None)
        if self.id is None:
            logging.debug(f"Creating a new qubit with ID {BaseQuantState.qubit_counter}")
            self.id = BaseQuantState.qubit_counter
            BaseQuantState.qubit_counter += 1
        self.history = kwargs.get("history", [])
        self.skip_val = kwargs.get("skip_val", False)
        
        self.state = kwargs.get("state", None)
        self.rho: list = kwargs.get("rho", None)
        self.state_type = None
        base_quant_state_validation(self)


    def log_history(self, message):
        if isinstance(message, str) and isinstance(self.id, int):
            self.history.append(message)
        else:
            raise BaseQuantumStateError(f"message cannot be of type {type(message)}, expected type str")
        
    def print_history(self):
        logging.info(f"Qubit {self.id} History")
        for entry in self.history:
            logging.info(f"- {entry}")

    def __str__(self: "BaseQuantState") -> str:
        rho = dense_mat(self.rho)
        rho_str = np.array2string(rho, precision=p_prec, separator=', ', suppress_small=True)
        if self.display_mode == "density":
            return f"Q{self.id}:\n{self.state_type}\n{rho_str}"
        elif self.display_mode == "ind_qub":
            return '\n'.join([f"Q{self.id} ({i}):\n{self[i].rho}" for i in range(self.n)])
        state_print = self.build_state_from_rho()
        if isinstance(state_print, tuple):
            raise BaseStatePreparationError(f"The state vector of a pure state cannot be a tuple")
        state_str = np.array2string(dense_mat(state_print), precision=p_prec, separator=', ', suppress_small=True)
        if self.display_mode == "vector":
            return f"Q{self.id}:\n{state_str}"
        elif self.display_mode == "both":
            return f"Q{self.id}:\nState:\n{state_str}\nRho:\n{rho_str}"
        
    def __setattr__(self: "BaseQuantState", name: str, value) -> None:
        if name == "immutable":
            object.__setattr__(self, name, True)
        if not hasattr(self, "_initialised"): 
            object.__setattr__(self, name, value)
            return
        if hasattr(self, "immutable") and name in self.immutable_attr:
            current_value = getattr(self, name, None)
            if name == "rho" or name == "state":
                if np.array_equal(dense_mat(current_value), dense_mat(value)) and self.class_type != "symbqubit":
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
        excluded_classes = ["symbqubit"]
        if self.class_type in excluded_classes:
            return
        purity = (self.rho.dot(self.rho)).diagonal().sum().real if sparse.issparse(self.rho) else np.einsum('ij,ji', self.rho, self.rho).real  
        if self.skip_val == True:
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
        """Builds a pure rho matrix, primarily in initiation of Qubit object"""
        if isinstance(self.state, sp.MatrixBase):
            return self.state * self.state.H
        elif len(self.state) > dense_limit:
            return sparse_array(self.state) @ sparse_array(self.state.conj().T)
        elif isinstance(self.state, np.ndarray):
            return np.einsum("i,j", np.conj(self.state), self.state, optimize=True)
        
    def set_display_mode(self: "BaseQuantState", mode: str) -> None:
        """Sets the display mode between the three options, returns type None"""
        if mode not in ["vector", "density", "both", "ind_qub"]:
            raise BaseQuantumStateError(f"The display mode must be set in 'vector', 'density' or 'both'")
        non_vector_state_types = ["Mixed", "Non-Unitary"]
        if self.state_type in non_vector_state_types and mode != "density":
            raise BaseQuantumStateError(f"Mixed and Non-Unitary states can only be displayed as density matrices")
        self.display_mode = mode
        
    def __repr__(self: "BaseQuantState") -> str:
        return self.__str__()

    def __copy__(self: "BaseQuantState") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem")
    
    def __deepcopy__(self: "BaseQuantState") -> None:
        raise BaseQuantumStateError(f"Qubits cannot be copied as decreed by the No-Cloning Theorem, its twice the sin to try to double copy them")
    
    def __mul__(self: "BaseQuantState", other: int | float) -> "BaseQuantState":
        if isinstance(other, (int, float)):
            new_rho = self.rho * other
            kwargs = {"rho": new_rho, "skip_val": True}
            kwargs.update(copy_quant_state_attr(self, kwargs))
            kwargs["history"].append(f"Multipled by {other}") if "history" in kwargs else None
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __rmul__(self: "BaseQuantState", other: int | float) -> "BaseQuantState":
        return self.__mul__(other)
    
    def __imul__(self: "BaseQuantState", other: float) -> "BaseQuantState":
        self.log_history(f"Multipled by {other}")
        if isinstance(other, (int, float)):
            self.rho *= other
            return self
        raise BaseQuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
    def __truediv__(self: "BaseQuantState", other: int | float) -> "BaseQuantState":
        if isinstance(other, (int, float)):
            new_rho = self.rho / other
            kwargs = {"rho": new_rho, "skip_val": True}
            kwargs.update(copy_quant_state_attr(self, kwargs))
            kwargs["history"].append(f"Divided by {other}") if "history" in kwargs else None
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")

    def __itruediv__(self: "BaseQuantState", other: float) -> "BaseQuantState":
        self.log_history(f"Divided by {other}")
        if isinstance(other, (int, float)):
            self.rho /= other
            return self
        raise BaseQuantumStateError(f"The variable with which you are multiplying the Qubit by cannot be of type {type(other)}, expected type int or type float")
    
    def __sub__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":
        """Subtraction of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, self.__class__):
            rho_1, rho_2 = auto_choose(self.rho, other.rho)
            new_rho = rho_1 - rho_2
            kwargs = {"rho": new_rho, "skip_val": True}                #CAREFUL skip val here
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            kwargs["history"].append(f"Subtracted by State {other.id}") if "history" in kwargs else None
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")
    
    def __isub__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":
        self.log_history(f"Subtracted by State {other.id}")
        logging.debug(f"Iteratively subtracting rho matrices")
        if isinstance(other, self.__class__):
            if hasattr(self, "immutable") and self.immutable:
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self - other
            return self
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")
    
    def __add__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":
        """Addition of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, self.__class__):
            rho_1, rho_2 = auto_choose(self.rho, other.rho)
            new_rho = rho_1 + rho_2
            kwargs = {"rho": new_rho, "skip_val": True}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            kwargs["history"].append(f"Added to State {other.id}") if "history" in kwargs else None
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")
    
    def __iadd__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":
        self.log_history(f"Added to State {other.id}")
        if isinstance(other, self.__class__):
            if hasattr(self, "immutable") and self.immutable:
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self + other
            return self
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")
    
    def __matmul__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":     
        """Matrix multiplication between two Qubit objects, returns a Qubit object"""
        if isinstance(other, self.__class__):
            rho_1, rho_2 = auto_choose(self.rho, other.rho)
            if sparse.issparse(rho_1):
                new_rho = rho_1 @ rho_2
            else:
                new_rho = np.dot(rho_1, rho_2)
            kwargs = {"rho": new_rho, "skip_val": True}             #not guarenteed to be hermitian
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            kwargs["history"].append(f"Matrix multipled with State {other.id}") if "history" in kwargs else None
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")

    def __imatmul__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":

        if isinstance(other, self.__class__):
            if hasattr(self, "immutable") and self.immutable:
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self @ other
            return self
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")
    
    def __mod__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":
        """Tensor product among two Qubit objects, returns a Qubit object"""
        if isinstance(other, self.__class__):
            rho_1, rho_2 = auto_choose(self.rho, other.rho, tensor=True)
            if sparse.issparse(rho_1):
                new_rho = sparse.kron(rho_1, rho_2)
            else:
                new_rho = np.kron(rho_1, rho_2)
            if self.history == []:
                kwargs = {"rho": new_rho, "history": [f"{self.id} tensored with {other.id}"]}
            else:
                kwargs = {"rho": new_rho}
            kwargs.update(combine_quant_state_attr(self, other, kwargs))
            kwargs["history"].append(f"Tensored with state {other.id}") if "history" in kwargs and self.history != [] else None
            return self.__class__(**kwargs)
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")

    def __imod__(self: "BaseQuantState", other: "BaseQuantState") -> "BaseQuantState":
        self.log_history(f"Tensored with State {other.id}")
        if isinstance(other, self.__class__):
            if hasattr(self, "immutable") and self.immutable:
                raise BaseQuantumStateError(f"This operation is not valid for an immutable object")
            self = self % other
            return self
        raise BaseQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected Qubits of the same types")

    def debug(self: "BaseQuantState", title=True) -> None:
        """Prints out lots of information on the Qubits core properties primarily for debug purposes, returns type None"""
        logging.info(f"\n")
        if title:
            logging.info("-" * linewid)
            logging.info(f"QUBIT DEBUG")
        if self.rho is not None:
            logging.info(f"self.rho.shape: {self.rho.shape}")
            logging.info(f"self.rho type: {type(self.rho)}")
            logging.info(f"self.rho:\n {self.rho}")
            logging.info(f"self.n: {self.n}")
            for i in range(self.n):
                logging.info(f"Qubit {i}: {self[i]}")
        logging.info(f"self.state:\n {self.build_state_from_rho()}") if self.rho is not None else print(f"self.state:\n {self.state}")
        logging.info(f"state_type: {self.state_type}")
        logging.info(f"All attributes and variables of the Qubit object:")
        logging.info(vars(self))
        logging.info(f"Qubit History: {self.print_history()}")
        if title:
            logging.info("-" * linewid)