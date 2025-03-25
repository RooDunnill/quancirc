import numpy as np
from ..circuit_utilities.circuit_errors import BitError


__all__ = ["Bit"]

class Bit:
    def __init__(self, num_bits: int, val: bool=False, **kwargs):
        if isinstance(val, (bool, int)):
            self.class_type = "bit"
            self.val = val
            if self.val not in range(0,2) and not isinstance(self.val, bool):
                raise BitError(f"self.val cannot be of {type(self.val)} and not be {0,1}. self.val: {self.val}")
            self.num_bits = num_bits
            bit_setup = np.ones(num_bits) if val == True else np.zeros(num_bits)
            self.bits = kwargs.get("bits", bit_setup)
            self.name = kwargs.get("name", None)
        else:
            raise BitError(f"val cannot be of type {type(val)}, expected type bool")
        if not isinstance(self.bits, (list, np.ndarray)):
            raise BitError(f"self.bits cannot be of type {type(self.bits)}, expected type np.ndarray or type list")

    def __str__(self):
        bits_int = self.bits.astype(int)
        return f"{self.name} bit array:\n {bits_int}" if self.name else f"Bit array:\n {bits_int}"
    
    def __setitem__(self, index: int, val: bool):
        if isinstance(index, int) and isinstance(val, (bool, int)):
            self.bits[index] = val
        else:
            raise BitError(f"Index and Val cannot be of type {type(index)} and {type(val)}, expected types int and (bool, int)")
    
    def __getitem__(self, index: int):
        if isinstance(index, int):
            return self.bits[index]
        elif isinstance(index, slice):
            return [self.bits[index] for bit in self.bits[index]]
        raise BitError(f"Index cannot be of type {type(index)}, expected inde of type int")
    
    def __len__(self):
        return len(self.bits)
    
    def __eq__(self, other):
        if isinstance(other, Bit):
            return self.bits == other.bits
        raise BitError(f"other cannot be of type {type(other)}, expected type Bit")

    def __and__(self, other):       #&
        if isinstance(other, Bit) and len(self) == len(other):
            return Bit(self.num_bits, bits=np.logical_and(self.bits[:], other.bits[:]))
        raise BitError(f"Cannot perform AND operation with object of type {type(other)}")
    
    def __or__(self, other):         #|
        if isinstance(other, Bit) and len(self) == len(other):
            return Bit(self.num_bits, bits=np.logical_or(self.bits[:], other.bits[:]))
        raise BitError(f"Cannot perform OR operation with object of type {type(other)}")
    
    def __xor__(self, other):        #^
        if isinstance(other, Bit) and len(self) == len(other):
            return Bit(self.num_bits, bits=np.logical_xor(self.bits[:], other.bits[:]))
        raise BitError(f"Cannot perform XOR operation with object of type {type(other)}")