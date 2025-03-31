import numpy as np
from ..circuit_config import *
from .circuit_errors import QC_error
from ..base_classes.base_qubit import BaseQubit

def binary_entropy(prob: float) -> float:
    """Used to calculate the binary entropy of two probabilities"""
    if isinstance(prob, (float, int)):
        if prob ==  0 or prob == 1:
            return 0.0
        else:
            return -prob*np.log2(prob) - (1 - prob)*np.log2(1 - prob)
    raise QC_error(f"Binary value must be a float")

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
