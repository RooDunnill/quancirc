from ..circuit_config import *
from .base_class_utilities.base_class_errors import BaseQuantumCircuitError
from .base_qubit import *
from .base_gate import *


class BaseCircuit:



    def set_display_mode(self, mode: str) -> None:
        """Sets the display mode between the three options, returns type None"""
        if mode not in ["vector", "density", "both"]:
            raise BaseQuantumCircuitError(f"The display mode must be set in 'vector', 'density' or 'both'")
        self.state.display_mode = mode

    def print_state(self, qubit=None) -> None:
        print(self.state[qubit]) if qubit else print(self.state)

    def return_state(self, qubit=None) -> BaseQubit:
        return self.state[qubit] if qubit else  self.state