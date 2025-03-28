import numpy as np
from ...circuit_utilities.circuit_errors import BitError


__all__ = ["Bit"]

class Bit:
    """Used to store classical bits, mainly as an output of measurements within the circuit class"""
    def __init__(self, string: str, **kwargs):
        self.class_type = "bit"
        self.bit_string = [int(b) for b in string]
        self.name = kwargs.get("name", None)
        self.verbose = kwargs.get("verbose", False)


    def __str__(self):
        return f"{self.name} bit array:\n{''.join(map(str, self.bit_string))}" if self.name else f"Bit array:\n{''.join(map(str, self.bit_string))}"
    
    def __setitem__(self, index: int, val: bool):
        """Allows for simple manipulation of bits"""
        if isinstance(index, int) and isinstance(val, (bool, int)):
            self.bit_string[index] = str(val)
        else:
            raise BitError(f"Index and Val cannot be of type {type(index)} and {type(val)}, expected types int and (bool, int)")
    
    def __getitem__(self, index: int):
        """Allows for simple and clear access to specific bits"""
        if isinstance(index, int):
            return self.bit_string[index]
        elif isinstance(index, slice):
            return [self.bit_string[index] for bit in self.bit_string]
        raise BitError(f"Index cannot be of type {type(index)}, expected inde of type int")
    
    def __len__(self):
        """retunrs the length of the bit string as the objects leangth attribute"""
        return len(self.bit_string)
    
    def __eq__(self, other):
        """Checks if two bit strings are identical or not"""
        if isinstance(other, Bit):
            return self.bit_string == other.bit_string
        raise BitError(f"other cannot be of type {type(other)}, expected type Bit")

    def __and__(self, other):       #&
        """The classical AND gate which acts on the bit string"""
        if isinstance(other, Bit) and len(self) == len(other):
            result = np.logical_and(self.bit_string[:], other.bit_string[:])
            return Bit("".join([str(bit) for bit in result.astype(int)]))
        raise BitError(f"Cannot perform AND operation with object of type {type(other)}")
    
    def __or__(self, other):         #|
        """The classical OR gate"""
        if isinstance(other, Bit) and len(self) == len(other):
            result = np.logical_or(self.bit_string[:], other.bit_string[:])
            return Bit("".join([str(bit) for bit in result.astype(int)]))
        raise BitError(f"Cannot perform OR operation with object of type {type(other)}")
    
    def __xor__(self, other):        #^
        """The classical XOR gate"""
        if isinstance(other, Bit) and len(self) == len(other):
            result = np.logical_xor(self.bit_string[:], other.bit_string[:])
            return Bit("".join([str(bit) for bit in result.astype(int)]))
        raise BitError(f"Cannot perform XOR operation with object of type {type(other)}")
    
    def __invert__(self):  #~
        """The classical NOT gate"""
        result = np.logical_not(self.bit_string[:])
        return Bit("".join([str(int(bit)) for bit in result.astype(int)]))
   
    def str_to_list(self, bit_str: str):
        """Turns the bit string into a list with eahc element a seperate entry"""
        return [int(b) for b in bit_str]
    
    def add_bits(self, bits):
        """Allows for the addition of bits, mostly due to measurements"""
        bit_list = [int(bit) for bit in bits]
        self.bit_string.extend(bit_list)

    def return_bits_as_str(self):
        """Converts the list back into a string"""
        return ''.join(map(str, self.bit_string))