import numpy as np
from .qubit import *
from .bit import *
from .gate import *
from .quant_info import *
from .measure import *
from utilities.qc_errors import QuantumCircuitError


class Circuit:
    def __init__(self, **kwargs):
        self.qubit_num = kwargs.get("q", 1)
        self.bit_num = kwargs.get("b", 1)
        self.gates = []
        self.state, self.bits = self.init_circuit()
        self.index_qubit = None
        self.final_gate = None
        self.prob_distribution = None
        self.circuit_gate = None

    def init_circuit(self) -> tuple[Qubit, Bit]:
        if isinstance(self.qubit_num, int):
            state = Qubit.q0(n=self.qubit_num)
        else:
            raise QuantumCircuitError(f"q cannot be of type {type(self.qubit_num)}, expected type int")
        if isinstance(self.bit_num, int):
            bits = Bit(self.bit_num)
        else:
            raise QuantumCircuitError(f"b cannot be of type {type(self.bit_num)}, expected type int")
        return state, bits

    def __str__(self):
        return f"{self.state}\n{self.circuit_gate}\n{self.prob_distribution}"
    
    def add_gate(self, gate, qubit=None):
        if qubit is not None:
            print("X" * 40)
            print(Gate.Identity(n=qubit))
            print(gate)
            print(Gate.Identity(n=self.qubit_num - qubit * gate.n - gate.n))
            print("X" * 40)
            enlarged_gate = Gate.Identity(n=qubit) @ gate @ Gate.Identity(n=self.qubit_num - qubit * gate.n - gate.n)
            self.gates.append(enlarged_gate)
        else:
            self.gates.append(gate)

    def compute_final_gate(self):
        self.final_gate = Gate.Identity(n=self.qubit_num)
        for gate in reversed(self.gates):         #goes backwards through the list and applies them
            self.final_gate = self.final_gate * gate
        return self.final_gate

    def apply_gates(self):
        if self.final_gate is None:
            self.final_gate = self.compute_final_gate()           #applies the function to multiply them all together
        self.state = self.final_gate * self.state
        self.circuit_gate = self.final_gate
        self.final_gate = None
        return self.state

    def list_probs(self, qubit=None, povm=None):
        if qubit is not None:
            self.prob_distribution = Measure(self.state[qubit]).list_probs(povm)            #just lists all of the probabilities of that computed state
        else:
            self.prob_distribution = Measure(self.state).list_probs(povm)
        return self.prob_distribution

    def measure_state(self, qubit=None, povm=None):
        if qubit is not None:
            self.state[qubit] = Measure(self.state[qubit]).measure_state(povm)
            return self.state
        else:
            return Measure(state=self.state).measure_state(povm)

    def get_info(self):
        pass

    def debug(self):
        pass

