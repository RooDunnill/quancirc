from ..circuit_config import *
from .base_class_utilities.base_class_errors import BaseQuantumCircuitError
from .base_qubit import *
from .base_gate import *


class BaseCircuit:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Quantum Circuit")



    def set_display_mode(self, mode: str) -> None:
        """Sets the display mode between the three options, returns type None"""
        if mode not in ["vector", "density", "both"]:
            raise BaseQuantumCircuitError(f"The display mode must be set in 'vector', 'density' or 'both'")
        self.state.display_mode = mode

    def print_state(self, index=0, qubit=None) -> None:
        print(self.qubit_array[index][qubit]) if qubit else print(self.qubit_array[index])

    def return_state(self, index=0, qubit=None) -> BaseQubit:
        return self.qubit_array[index][qubit] if qubit else  self.qubit_array[index]