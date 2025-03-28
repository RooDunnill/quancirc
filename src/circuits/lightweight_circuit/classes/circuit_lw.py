import numpy as np
from .qubit_lw import *
from ...circuit_utilities import *
from scipy.sparse import eye_array
from ...base_circuit.classes import *
from ...circuit_config import linewid


__all__ = ["Circuit_LW"]

class Circuit_LW:
    """A circuit build out of 1D arrays that can only handle pure states, howver is optimised to be overlla faster than the base circuit"""
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'circuit_lw')
        self.qubit_num = kwargs.get("q", 1)
        self.bit_num = kwargs.get("b", 1)
        self.verbose = kwargs.get("verbose", True)
        self.gates = []
        self.depth = 0
        self.index_qubit = None
        self.collapsed_qubits = []
        self.collapsed = False
        self.state, self.bits = self.init_circuit()


    def init_circuit(self) -> tuple[Qubit, Bit]:
        if self.verbose:
            print("\n")
            print("=" * linewid)
            print(f"Initialising circuit with {self.qubit_num} qubits and {self.bit_num} bits")
        return Qubit_LW.q0(n=self.qubit_num), Bit("00000000")
    

    def apply_gate(self, gate, qubit=None, **kwargs) -> None:
        gate_name = gate.name
        if self.collapsed:
            raise QuantumCircuitError(f"This state has already been measured and so no further gates can be applied")
        if gate is Identity:
            if self.verbose:
                print(f"Applying {gate.name} to qubit {qubit}")
            return
        fwht = kwargs.get("fwht", False)
        if gate is not Hadamard and fwht == True:
            raise LWQuantumCircuitError(f"fwht can only be used when the gate is Hadamard")
        if gate is Hadamard and qubit is None and fwht == True:
            self.state = vector_fwht(self.state)
            if self.verbose:
                print(f"Applying Fast Walsh Hadamard Transform to the state")
        else:
            if qubit is not None:        #MAKE SO IT APPLIES JUST TO THAT QUBIT AND THEN RETENSORS
                if qubit in self.collapsed_qubits:
                    raise QuantumCircuitError(f"A gate cannot be applied to qubit {qubit}, as it has already been measured and collapsed")
                self.state.state = sparse_array(self.state.state)    #this is changing the shape i believe FIXXXXXXXXXXXXXXXX
                gate_action = Gate(matrix=sparse_mat(gate.matrix))
                gate = Gate.Identity(n=qubit, type="sparse") % gate_action % Gate.Identity(n=self.state.n - qubit - 1, type="sparse")
                gate.name = f"{gate_name}{qubit}"
                self.state = gate @ self.state
                if self.verbose:
                    print(f"Applying {gate.name} to qubit {qubit}")
            elif qubit is None:
                self.state = gate @ self.state
                if self.verbose:
                    print(f"Adding {gate.name} of size {gate.n} x {gate.n} to the circuit")

    def list_probs(self, qubit=None):
        if qubit is None:
            if sparse.issparse(self.state.state):
                self.prob_distribution = self.state.state.multiply(self.state.state.conjugate())
            else:
                self.prob_distribution = np.einsum("i,i->i", self.state.state.ravel(), np.conj(self.state.state.ravel()))
            self.prob_distribution = dense_mat(self.prob_distribution).ravel().real
        if self.verbose:
            print(f"Listing the probabilities:\n{format_ket_notation(self.prob_distribution)}")
        return self.prob_distribution

    def measure_state(self, qubit=None):
        self.depth += 1
        if qubit is None:
            probs = self.list_probs()
            measurement = choices(range(len(probs)), weights=probs)[0]
            post_measurement_vector = np.zeros((self.state.dim), dtype=np.complex128)
            post_measurement_vector[measurement] = 1
            kwargs = {"state": post_measurement_vector}
            kwargs.update(copy_qubit_attr(self))
            self.collapsed = True
            if self.verbose:
                print(f"Measured the state {measurement} of the whole system")
            return Qubit(**kwargs)
        
    def get_info(self):
        return QuantInfo.state_info(self.state)
    
    def print_state(self):
        print(self.state)

    def return_state(self):
        return self.state

    def purity(self, qubit=None):
        purity = QuantInfo.purity(self.state[qubit]) if qubit else QuantInfo.purity(self.state)
        if self.verbose:
            print(f"Purity of the qubit {qubit} is {purity}") if qubit else print(f"Purity of the state is {purity}")
        return purity
    
    def linear_entropy(self, qubit=None):
        linear_entropy = QuantInfo.linear_entropy(self.state[qubit]) if qubit else QuantInfo.linear_entropy(self.state)
        if self.verbose:
            print(f"Linear Entropy of the qubit {qubit} is {linear_entropy}") if qubit else print(f"Linear Entropy of the state is {linear_entropy}")
        return linear_entropy
    
    def vn_entropy(self, qubit=None):
        return QuantInfo.vn_entropy(self.state[qubit]) if qubit else QuantInfo.vn_entropy(self.state)
    
    def shannon_entropy(self, qubit=None):
        return QuantInfo.shannon_entropy(self.state[qubit]) if qubit else QuantInfo.shannon_entropy(self.state)

    def debug(self, title=True):
        print("-" * linewid)
        print(f"CIRCUIT DEBUG")
        print(f"Circuit Depth: {self.depth}")
        print(f"Number of Qubits: {self.qubit_num}")
        print(f"Number of Bits: {self.bit_num}")
        print(f"Collapsed State: {self.collapsed}")
        print(f"Collapsed Qubits: {self.collapsed_qubits}")
        print(f"")
        print(f"\nCircuit State Debug Information:")
        print("-" * (int(linewid/2)))
        self.state.debug(title=False)
        print("-" * linewid)