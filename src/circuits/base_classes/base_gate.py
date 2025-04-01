import numpy as np
from ..circuit_config import *
from  ..circuit_utilities.sparse_funcs import *
from .base_class_utilities.base_class_errors import BaseGateError


def combine_gate_attr(self: "BaseGate", other: "BaseGate", op = "+") -> list:
        """Allows the returned objects to still return name too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            if op == "%":
                new_name = f"{self.name}{other.name}"
            elif op:
                new_name = f"{self.name} {op} {other.name}"
            if len(new_name) > name_limit:
                new_name = new_name[len(new_name) - name_limit:]
            kwargs["name"] = new_name
        if hasattr(self, "skip_val") and self.skip_val == True:
            kwargs["skip_validation"] = True
        elif hasattr(other, "skip_val") and other.skip_val == True: 
            kwargs["skip_validation"] = True
        return kwargs

class BaseGate:
    def __init__(self, **kwargs):
        self.skip_val = kwargs.get("skip_validation", False)
        self.name = kwargs.get("name", "QG")
        self.matrix = kwargs.get("matrix", None)

    def __str__(self) -> str:
        matrix_str = np.array2string(dense_mat(self.matrix), precision=p_prec, separator=', ', suppress_small=True)
        return f"{self.name}\n{matrix_str}"

    def __setattr__(self: "BaseGate", name: str, value) -> None:
        if name == "immutable":
            object.__setattr__(self, name, True)
        if not hasattr(self, "_initialised"): 
            object.__setattr__(self, name, value)
            return
        if hasattr(self, "immutable") and name in self.immutable_attr:
            current_value = getattr(self, name, None)
            if name == "rho" or name == "state":
                if np.array_equal(dense_mat(current_value), dense_mat(value)):
                    object.__setattr__(self, name, value)
                    return
            elif current_value == value:
                return
            raise AttributeError(f"Cannot modify immutable object: {name}")
        if name in self.all_immutable_attr:
            raise AttributeError(f"Cannot modify immutable object: {name}")
        object.__setattr__(self, name, value)

    def __getitem__(self, index: tuple[int, int]):
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.matrix[row, col]
        
    def __mul__(self: "BaseGate", other: int | float) -> "BaseGate":
        if isinstance(other, (int, float)):
            new_matrix = self.matrix * other
            kwargs = {"matrix": new_matrix, "skip_validation": True}
            kwargs.update(combine_gate_attr(self, other, op = "*"))
            return self.__class__(**kwargs)
        raise QuantumStateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")

    def __rmul__(self: "BaseGate", other: int | float) -> "BaseGate":
        return self.__mul__(other)
    
    def __imul__(self: "BaseGate", other: float) -> "BaseGate":
        if isinstance(other, (int, float)):
            self.matrix *= other
            return self
        raise QuantumStateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")
    
    def __truediv__(self: "BaseGate", other: int | float) -> "BaseGate":
        if isinstance(other, (int, float)):
            new_matrix = self.matrix / other
            kwargs = {"matrix": new_matrix, "skip_validation": True}
            kwargs.update(combine_gate_attr(self, other, op = "*"))
            return self.__class__(**kwargs)
        raise BaseGateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")

    def __itruediv__(self: "BaseGate", other: float) -> "BaseGate":
        if isinstance(other, (int, float)):
            self.matrix /= other
            return self
        raise BaseGateError(f"The variable with which you are multiplying the Gate by cannot be of type {type(other)}, expected type int or type float")
    
    def __sub__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        """Subtraction of two Qubit rho matrices, returns a Gate object"""
        if isinstance(other, BaseGate):
            mat_1 = convert_to_sparse(self.matrix)
            mat_2 = convert_to_sparse(other.matrix)
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = sparse_mat(self.matrix) - sparse_mat(other.matrix)
            else:
                new_matrix = dense_mat(self.matrix) - dense_mat(other.matrix)
            kwargs = {"matrix": new_matrix, "skip_validation": True}                #CAREFUL skip val here
            kwargs.update(combine_gate_attr(self, other, op = "-"))
            return self.__class__(**kwargs)
        raise BaseGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __isub__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        if isinstance(other, BaseGate):
            if hasattr(self, "immutable") and self.immutable:
                raise BaseGateError(f"This operation is not valid for an immutable object")
            self = self - other
            return self
        raise BaseGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __add__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        """Addition of two Qubit rho matrices, returns a Qubit object"""
        if isinstance(other, BaseGate):
            mat_1 = convert_to_sparse(self.matrix)
            mat_2 = convert_to_sparse(other.matrix)
            if sparse.issparse(mat_1) and sparse.issparse(mat_2):
                new_matrix = sparse_mat(self.matrix) + sparse_mat(other.matrix)
            else:
                new_matrix = dense_mat(self.matrix) + dense_mat(other.matrix)
            kwargs = {"matrix": new_matrix, "skip_validation": True}
            kwargs.update(combine_gate_attr(self, other, op = "+"))
            return self.__class__(**kwargs)
        raise BaseGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __iadd__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        if isinstance(other, BaseGate):
            if hasattr(self, "immutable") and self.immutable:
                raise BaseGateError(f"This operation is not valid for an immutable object")
            self = self + other
            return self
        raise BaseGateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")

    def __imatmul__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        if isinstance(other, BaseGate):
            if hasattr(self, "immutable") and self.immutable == True:
                raise BaseGateError(f"This operation is not valid for an immutable object")
            self = self @ other
            return self
        raise BaseGateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected types Gate and Gate")

    def __imod__(self: "BaseGate", other: "BaseGate") -> "BaseGate":
        if isinstance(other, BaseGate):
            if hasattr(self, "immutable") and self.immutable == True:
                raise BaseGateError(f"This operation is not valid for an immutable object")
            self = self % other
            return self
        raise BaseGateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected types Gate and Gate")