import numpy as np
from .qubit import *
from .bit import *
from .gate import *
from .quant_info import *
from .measure import *
from utilities.qc_errors import QuantumCircuitError
from utilities.validation_funcs import circuit_validation


    



class Circuit:
    def __init__(self, **kwargs):
        self.qubit_num = kwargs.get("q", 1)
        self.bit_num = kwargs.get("b", 1)
        self.verbose = kwargs.get("verbose", False)
        circuit_validation(self)
        self.gates = []
        self.index_qubit = None
        self.prob_distribution = None
        self.circuit_gate = None
        self.qubits_to_bits = []
        self.collapsed = False
        self.state, self.bits = self.init_circuit()
    

    def init_circuit(self) -> tuple[Qubit, Bit]:
        return Qubit.q0(n=self.qubit_num), Bit(self.bit_num)

    def __str__(self):
        return f"{self.state}\n{self.prob_distribution}"
    
    def add_gate(self, gate, qubit=None):
        if self.collapsed:
            raise QuantumCircuitError(f"This state has already been measured and so no further gates can be applied")
        if qubit is not None:
            if qubit in self.qubits_to_bits:
                raise QuantumCircuitError(f"A gate cannot be applied to qubit {qubit}, as it has already been measured and collapsed")
            enlarged_gate = Gate.Identity(n=qubit) @ gate @ Gate.Identity(n=self.qubit_num - qubit * gate.n - gate.n)
            self.gates.append(enlarged_gate)
        else:
            self.gates.append(gate)
        

    def compute_final_gate(self):
        final_gate = Gate.Identity(n=self.qubit_num)
        for gate in reversed(self.gates):         #goes backwards through the list and applies them
            final_gate = final_gate * gate
        return final_gate

    def apply_gates(self):
        for gate in self.gates:
            self.state = gate * self.state
            self.state.set_display_mode("both")
        self.gates = []
        return self.state

    def list_probs(self, qubit=None, povm=None):
        self.prob_distribution = Measure(self.state if qubit is None else self.state[qubit]).list_probs(povm)
        return self.prob_distribution

    def measure_state(self, qubit=None, povm=None):
        if qubit is not None:
            self.state[qubit] = Measure(self.state[qubit]).measure_state(povm)
            self.qubits_to_bits.append(qubit)
            return self.state
        else:
            self.state = Measure(state=self.state).measure_state(povm)
            self.collapsed = True
            return self.state
        
    def purity(self):
        return QuantInfo.purity(self.state)
    
    def linear_entropy(self):
        return QuantInfo.linear_entropy(self.state)

    def get_info(self):
        return QuantInfo.state_info(self.state)
    
    def vn_entropy(self):
        return QuantInfo.vn_entropy(self.state)
    
    def shannon_entropy(self):
        return QuantInfo.shannon_entropy(self.state)

    def debug(self):
        pass
