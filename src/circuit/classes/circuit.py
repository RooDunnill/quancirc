import numpy as np
from .qubit import *
from .bit import *
from .gate import *
from .quant_info import *
from .measure import *
from .qubit_array import *
from .circuit_special_gates import *
from ..circuit_utilities import *
from ..circuit_config import *
from scipy.sparse import eye_array

__all__ = ["Circuit"]
        
class Circuit:
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'circuit')
        self.qubit_num = kwargs.get("q", 1)
        self.bit_num = kwargs.get("b", 1)
        self.verbose = kwargs.get("verbose", True)
        circuit_validation(self)
        self.gates = []
        self.depth = 0
        self.index_qubit = None
        self.prob_distribution = None
        self.circuit_gate = None
        self.collapsed_qubits = []
        self.collapsed = False
        self.circuit_mode = kwargs.get("mode", "circuit")
        self.state, self.bits = self.init_circuit()
        self.qubit_array = None

    def init_circuit(self) -> tuple[Qubit, Bit]:
        if self.verbose:
            if self.circuit_mode == "circuit":
                print("\n")
                print("=" * linewid)
                print(f"Curcuit mode set to 'circuit', will now create the qubits and bits for the state")
                print(f"Initialising circuit with {self.qubit_num} qubits and {self.bit_num} bits")
                return Qubit.q0(n=self.qubit_num), Bit("", verbose=self.verbose)
            elif self.circuit_mode == "array":
                print("\n")
                print("=" * linewid)
                print(f"Circuit mode set to 'array', will await array upload")
                return Qubit.q0(n=1), Bit("", verbose=self.verbose)
    
    def config_noise(self, **kwargs):
        self.noisy = kwargs.get("noise", False)
        self.noise_type = kwargs.get("noise_type", None)
        self.channel = kwargs.get("channel", None)


    def __dir__(self):
        methods = ["upload_qubit_array", "apply_gate", "list_probs", "measure_state", 
                   "get_info", "print_states", "return_states", "purity", "linear_entropy", "vn_entropy",
                   "shannon_entropy", "apply_channel_to_qubit", "apply_local_channel_to_qubit", "debug", "return_bits"]
        if self.qubit_array:
            methods.extend(["download_qubit_array", "apply_gate_on_array", "get_array_info"])
            methods.remove(["apply_gate", "list_probs", "measure_state","apply_channel_to_qubit", "apply_local_channel_to_qubit",
                            "return_state", "print_state", "purity", "linear_entropy", "vn_entropy", "shannon_entropy"])
        return methods
    
    def __getattr__(self, name):
        if name not in self.__dir__():
            raise QuantumCircuitError(f"function cannot be used currently")
            
    def __str__(self):
        return f"{self.state}\n{self.prob_distribution}"
    
    def __getitem__(self, index):
        if index < len(self.qubit_array):
            if self.verbose:
                print(f"Retreiving qubit {index} from the qubit array")
            return self.qubit_array[index]
        else:
            raise QuantumCircuitError(f"Cannot get a qubit from index {index}, when the array length is {len(self.qubit_array)}")
        
    def __setitem__(self, index, qub):
        if index < len(self.qubit_array):
            if isinstance(qub, Qubit):
                self.qubit_array[index] = qub
            else:
                raise QuantumCircuitError(f"The inputted value cannot be of type {type(qub)}, expected type Qubit")
        else:
            raise QuantumCircuitError(f"Cannot asign a qubit to index {index}, when the array length is {len(self.qubit_array)}")
    
    def upload_qubit_array(self, qubit_arr: QubitArray) -> None:
        if self.circuit_mode == "circuit":
            print(f"Switching circuit mode to 'array'") if self.verbose else None
        if self.qubit_array is None:
            print(f"Uploading qubit array...") if self.verbose else None
            self.qubit_array = qubit_arr.qubit_array
            print(f"Upload Complete") if self.verbose else None
        else:
            raise QuantumCircuitError(f"Please download current qubit array before uploading the new qubit array, {qubit_arr.name}")

    def download_qubit_array(self) -> QubitArray:
        if len(self.qubit_array) != 0:
            print(f"Downloading qubit array...") if self.verbose else None
            qubit_array = QubitArray(array=self.qubit_array)
            print(f"Download Complete") if self.verbose else None
            return qubit_array
        else:
            raise QuantumCircuitError(f"There is no qubit array to download currently in the circuit")

    def return_bits(self):
        return self.bits

    def apply_gate_on_array(self, gate: Gate, index, qubit=None, verbose=True):
        if gate is Identity:
            if self.verbose and verbose:
                print(f"Applying {gate.name} to qubit {index}")
            return
        if qubit:
            gate = Gate.Identity(n=qubit, type="dense") % gate % Gate.Identity(n=self.state.n - qubit - 1, type="dense")
            self.qubit_array[index] = gate @ self.qubit_array[index]
            if self.verbose and verbose:
                print(f"Applying {gate.name} of size {gate.n} x {gate.n} to the qubit number {index} to qubit ")
        else:
            self.qubit_array[index] = gate @ self.qubit_array[index]
            if self.verbose and verbose:
                print(f"Applying {gate.name} of size {gate.n} x {gate.n} to the qubit number {index} to the whole state")
        
    def apply_gate(self, gate, qubit=None) -> None:
        gate_name = gate.name
        if self.collapsed:
            raise QuantumCircuitError(f"This state has already been measured and so no further gates can be applied")
        if gate is Identity:
            if self.verbose:
                print(f"Applying {gate.name} to qubit {qubit}")
            return
        if qubit is not None:       
            if qubit in self.collapsed_qubits:
                raise QuantumCircuitError(f"A gate cannot be applied to qubit {qubit}, as it has already been measured and collapsed")
            self.state.rho = sparse_mat(self.state.rho)
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

    def list_probs(self, qubit=None, povm=None):
        self.prob_distribution = Measure(self.state if qubit is None else self.state[qubit]).list_probs(povm)
        if self.verbose:
            print(f"Listing the probabilities:\n{format_ket_notation(self.prob_distribution)}")
        return self.prob_distribution
    
    def measure_states_on_array(self, index, basis=None, povm=None):
        if index == "all":
            qubit_size = self.qubit_array[0].n
            for i in range(len(self.qubit_array)):
                if basis == "Z":
                    print(f"Measuring all qubits in the Z basis") if self.verbose else None
                    self.qubit_array[i] = Measure(self.qubit_array[i]).measure_state()
                elif basis == "X":
                    print(f"Measuring all qubits in the X basis") if self.verbose else None
                    self.apply_gate_on_array(Hadamard, i)
                    self.qubit_array[i] = Measure(self.qubit_array[i]).measure_state()
                    self.apply_gate_on_array(Hadamard, i)
                elif basis == "Y":
                    print(f"Measuring all qubits in the Y basis") if self.verbose else None
                    self.apply_gate_on_array(Gate.Rotation_Y(np.pi/2), i)
                    self.qubit_array[i] = Measure(self.qubit_array[i]).measure_state()
                    self.apply_gate_on_array(Gate.Rotation_Y(np.pi/2), i)
                elif povm:
                    self.qubit_array[i].state = Measure(self.qubit_array[i]).measure_state(povm)
        elif isinstance(index, int):
            if basis == "Z":
                print(f"Measuring qubit number {index} in the Z basis") if self.verbose else None
                self.qubit_array[index] = Measure(self.qubit_array[index]).measure_state()
                measured_state = np.argmax(np.diag(self.qubit_array[index].rho))
            elif basis == "X":
                print(f"Measuring qubit number {index} in the X basis") if self.verbose else None
                self.apply_gate_on_array(Hadamard, index, verbose=False)
                self.qubit_array[index] = Measure(self.qubit_array[index]).measure_state()
                measured_state = np.argmax(np.diag(self.qubit_array[index].rho))
                self.apply_gate_on_array(Hadamard, index, verbose=False)
            elif basis == "Y":
                print(f"Measuring qubit number {index} in the Y basis") if self.verbose else None
                self.apply_gate_on_array(Gate.Rotation_Y(np.pi/2), index, verbose=False)
                self.qubit_array[index] = Measure(self.qubit_array[index]).measure_state()
                measured_state = np.argmax(np.diag(self.qubit_array[index].rho))
                self.apply_gate_on_array(Gate.Rotation_Y(np.pi/2), index, verbose=False)
            elif povm:
                self.qubit_array[index] = Measure(self.qubit_array[index]).measure_state(povm)
            if measured_state is not None:
                self.bits.add_bits(str(measured_state))

    def measure_state(self, qubit=None, povm=None):
        self.depth += 1
        if qubit is not None:
            measurement = Measure(self.state[qubit]).measure_state(povm)
            self.state[qubit] = measurement
            self.collapsed_qubits.append(qubit)
            if self.verbose:
                measurement.set_display_mode("density")
                print(f"Measured the state {measurement} of qubit {qubit}")
            return self.state
        else:
            measurement = Measure(state=self.state).measure_state(povm)
            self.state = measurement
            self.collapsed = True
            if self.verbose:
                print(f"Measured the state {measurement} of the whole system")
            return self.state
        
    def get_info(self):
        return QuantInfo.state_info(self.state)
    
    def get_array_info(self):
        for i in range(len(self.qubit_array)):
            QuantInfo.qubit_info(self.qubit_array[i])

    
    def print_state(self, qubit=None):
        if qubit:
            print(self.state[qubit])
        else:
            print(self.state)

    def return_state(self, qubit=None):
        if qubit:
            return self.state[qubit]
        else:
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
    

    def kraus_generator(self, channel, prob):
        K0 = Gate.Identity(n = self.state.n)
        K0 *= np.sqrt(1 - prob)**self.state.n
        if channel == "Depol":
            Kx = X_Gate
            Ky = Y_Gate
            Kz = Z_Gate
            for i in range(self.state.n - 1):
                Kx = Kx % X_Gate
                Ky = Ky % Y_Gate
                Kz = Kz % Z_Gate
            Kx *= np.sqrt(prob/3)**self.state.n
            Ky *= np.sqrt(prob/3)**self.state.n
            Kz *= np.sqrt(prob/3)**self.state.n
            return K0, Kx, Ky, Kz
        elif channel == "X":
            K1 = X_Gate
        elif channel == "Y":
            K1 = Y_Gate
        elif channel == "Z":
            K1 = Z_Gate
        gate = K1
        for i in range(self.state.n - 1):
            K1 = K1 % gate
        K1 *= np.sqrt(prob)**self.state.n
        return K0, K1
    
    def single_kraus_generator(self, channel, prob):
        K0 = Gate.Identity()
        K0.skip_val = True
        K0 *= np.sqrt(1 - prob)
        if channel == "Depol":
            Kx = Gate.X_Gate()
            Kx.skip_val = True
            Ky = Gate.Y_Gate()
            Ky.skip_val = True
            Kz = Gate.Z_Gate()
            Kz.skip_val = True
            return K0, Kx, Ky, Kz
        elif channel == "X":
            K1 = Gate.X_Gate()
        elif channel == "Y":
            K1 = Gate.Y_Gate()
        elif channel == "Z":
            K1 = Gate.Z_Gate()
        K1.skip_val = True
        K1 *= np.sqrt(prob)
        return K0, K1
    
    def apply_channel_to_qubit(self, qubit, channel, prob):
        if self.collapsed:
            QuantumCircuitError(f"Cannot apply a quantum channel to a collapsed state")
        qubit_state = self.state[qubit]
        old_index = qubit_state.index
        old_name = qubit_state.name
        kraus_ops = self.single_kraus_generator(channel, prob)
        kraus_validation(kraus_ops)
        epsilon_rho = np.zeros((2, 2), dtype=np.complex128)
        epsilon = Qubit(rho=epsilon_rho, skip_validation=True)
        qubit_state.skip_val = True
        for k in kraus_ops:
            k_applied = k @ qubit_state
            epsilon += k_applied
        kwargs = {"rho": epsilon.rho, "skip_validation": False, "name": f"{channel} channel applied to {old_name}", "index": old_index}
        qubit_state = Qubit(**kwargs)
        self.state[qubit] = qubit_state
        qubit_state.set_state_type()
        self.state.state_type = qubit_state.state_type
        return self.state

    def apply_local_channel_to_state(self, channel, prob):
        for i in range(self.qubit_num):
            self.apply_channel_to_qubit(i, channel, prob)
        return self.state

    def apply_state_wide_channel(self, channel, prob):
        if self.collapsed:
            QuantumCircuitError(f"Cannot apply a quantum channel to a collapsed state")
        old_index = self.state.index
        old_name = self.state.name
        kraus_ops = self.kraus_generator(channel, prob)
        epsilon_rho = np.zeros((self.state.dim, self.state.dim), dtype=np.complex128)
        epsilon = Qubit(rho=epsilon_rho, skip_validation=True)
        self.state.skip_val = True
        for k in kraus_ops:
            k_applied = k @ self.state
            epsilon += k_applied
        kwargs = {"rho": epsilon.rho, "skip_validation": False, "name": f"{channel} channel applied to {old_name}", "index": old_index}
        self.state = Qubit(**kwargs)
        return self.state

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
