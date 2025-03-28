import numpy as np
from .qubit import *
from .quant_info import QuantInfo
from ...circuit_utilities.circuit_errors import QubitArrayError
from ...circuit_utilities.validation_funcs import qubit_array_validation

__all__ = ["QubitArray"]

class QubitArray:
    def __init__(self, q=1, qubit_size=1, **kwargs):
        self.name = kwargs.get("name", "Quantum Array")
        self.qubit_array = kwargs.get("array", [Qubit.q0(n=qubit_size)] * q)
        self.length = len(self)
        qubit_array_validation(self)

    def __len__(self):
        return len(self.qubit_array)

    def __getitem__(self, index: int):
        return self.qubit_array[index]
    
    def __setitem__(self, index: int, qub: Qubit):
        if isinstance(qub, Qubit):
            self.qubit_array[index] = qub
        else:
            raise QubitArrayError(f"The inputted value cannot be of type {type(qub)}, expected type Qubit")
        
    def set_array_length(self, length: int, delete=False):
        if length == len(self):
            return
        elif length >= len(self):
            self.qubit_array += [None] * (length - len(self))
        elif length <= len(self):
            if delete:
                self.qubit_array = self.qubit_array[:length]

    def add_qubit(self, qub: Qubit):
        self.qubit_array.append(qub)

    def insert_qubit(self, qub: Qubit, index):
        self.qubit_array.insert(index, qub)    
    
    def remove_qubit(self, index):
        del self.qubit_array[index]

    def pull_qubit(self, index):
        pulled_qubit = self.qubit_array.pop(index)
        return pulled_qubit
    
    def qubit_info(self, index):
        if len(self.qubit_array[index].state) == 2:
            return QuantInfo.qubit_info(self.qubit_array[index])
        return QuantInfo.state_info(self.qubit_array[index])
    
    def qubit_sizes(self):
        size_list = []
        for i in len(self):
            size_list.append(i.n)
        return size_list
    
    def pop_first_qubit(self):
        return self.qubit_array.pop(0)
    
    def validate_array(self):
        qubit_array_validation(self)


    