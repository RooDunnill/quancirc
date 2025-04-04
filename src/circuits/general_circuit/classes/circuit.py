import numpy as np
from ...circuit_config import *
from ...base_classes.base_circuit import *
from .qubit import *
from .bit import *
from .gate import *
from .quant_info import *
from .measure import *
from .qubit_array import *
from ..utilities.fwht import *
from ..utilities.validation_funcs import circuit_validation, kraus_validation
from ...circuit_utilities.sparse_funcs import *
from ...circuit_utilities.layout_funcs import *
from ..utilities.circuit_errors import QuantumCircuitError


__all__ = ["Circuit"]
        
class Circuit(BaseCircuit):
    """The main circuit in the program, allows for sparse and dense manipulation of 'full' qubits in rho form"""
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'circuit')
        self.states = kwargs.get("states", 0)
        self.qubit_num = kwargs.get("q", 1)
        self.bit_num = kwargs.get("b", 1)
        self.verbose = kwargs.get("verbose", True)
        circuit_validation(self)
        self.prob_distribution = None
        self.qubit_array = self.init_circuit()
        self.bits = Bit("", verbose=self.verbose)

    def init_circuit(self) -> tuple[Qubit, Bit]:
        """initialises the Quantum State and Bits for the circuit and prints initiation messages"""
        if self.states == 0:
            return []
        gen_qubits = QubitArray(q=self.states, qubit_size=self.qubit_num)
        return gen_qubits.qubit_array

            
    def __str__(self):
        return f"{self.qubit_array}\n{self.prob_distribution}" if self.prob_distribution is not None else f"{self.qubit_array}"
    
    def __getitem__(self, index: int) -> list:
        """Gets the qubit of that index  of the qubit array and returns"""
        if index < len(self.qubit_array):
            if self.verbose:
                print(f"Retreiving qubit {index} from the qubit array:")
            return self.qubit_array[index]
        else:
            raise QuantumCircuitError(f"Cannot get a qubit from index {index}, when the array length is {len(self.qubit_array)}")

    
    def __setitem__(self, index: int, qub: Qubit) -> None:
        """Sets the qubit of that index on the qubit array with whatever val you enter"""
        if index < len(self.qubit_array):
            if isinstance(qub, Qubit):
                self.qubit_array[index] = qub
            else:
                raise QuantumCircuitError(f"The inputted value cannot be of type {type(qub)}, expected type Qubit")
        else:
            raise QuantumCircuitError(f"Cannot asign a qubit to index {index}, when the array length is {len(self.qubit_array)}")

    
    def upload_qubit_array(self, qubit_arr: QubitArray) -> None:
        """Places the list of qubits within the object QubitArray into the circuit for gates and measurements to act upon them"""
        print(f"Uploading qubit array...") if self.verbose else None
        self.qubit_array.extend(qubit_arr.qubit_array)
        print(f"Upload Complete") if self.verbose else None

    def download_qubit_array(self, index=None) -> QubitArray:
        """Removes the qubit list from the circuit and creates a new QubitArray object"""
        if isinstance(index, int):
            qubit_array= QubitArray(array=self.qubit_array[index])
            self.qubit_array.pop(index)
            return qubit_array
        if index is None:
            print(f"Downloading qubit array...") if self.verbose else None
            qubit_array = QubitArray(array=self.qubit_array)
            self.qubit_array = None
            print(f"Download Complete") if self.verbose else None
            return qubit_array
        else:
            raise QuantumCircuitError(f"There is no qubit array to download currently in the circuit")

    def download_bits(self) -> Bit:
        """Returns the bits within the quantum circuit"""
        return self.bits
    
    def apply_fwht(self, index=0, verbose=True):
        print(f"Applying the FWHT to the state") if self.verbose and verbose else None
        self.qubit_array[index].rho = matrix_fwht(self.qubit_array[index].rho)
        
    def apply_gate(self, gate: Gate, index: int | range | list=0, qubit=None) -> None:
        """applies a gate to the quantum state when in normal mode"""
        gate_name = gate.name
        if isinstance(index, range):
            indices = list(index) 
        elif isinstance(index, int): 
            indices = [index]  
        elif isinstance(index, list): 
            indices = index
        for i in indices:
            if qubit is not None:       
                gate = Gate.Identity(n=qubit) % gate % Gate.Identity(n=self.qubit_array[i].n - qubit - 1)
                gate.name = f"{gate_name}{qubit}"
                self.qubit_array[i] = gate @ self.qubit_array[i]
                if self.verbose:
                    print(f"Applying {gate.name} to qubit {qubit}")
            elif qubit is None:
                if self.verbose:
                    print(f"Applying {gate.name} of size {gate.n} x {gate.n} to the circuit")
                self.qubit_array[i] = gate @ self.qubit_array[i]
            
    def list_probs(self, index: int=0, qubit=None, povm=None) -> np.ndarray:
        """lists the probabilities of the given state, can be applied to individual qubits"""
        self.prob_distribution = Measure(self.qubit_array[index] if qubit is None else self.qubit_array[index][qubit]).list_probs(povm)
        if self.verbose:
            print(f"Listing the probabilities:\n{format_ket_notation(self.prob_distribution)}")
        return self.prob_distribution
    
    def measure_state(self, index=0, qubit=None, basis=None, povm=None) -> None:
        if isinstance(index, range):
            indices = list(index) 
        elif isinstance(index, int): 
            indices = [index]  
        elif isinstance(index, list): 
            indices = index
        for i in indices:
            if basis == "Z":
                print(f"Measuring in the Z basis") if self.verbose else None
            elif basis == "X":
                print(f"Measuring in the X basis") if self.verbose else None
                self.apply_gate(Hadamard, index=i, qubit=qubit)
            elif basis == "Y":
                print(f"Measuring in the Y basis") if self.verbose else None
                self.apply_gate(Gate.Rotation_Y(np.pi/2), index=i, qubit=qubit)

            if qubit is not None:
                measurement = Measure(self.qubit_array[i][qubit]).measure_state(povm)
                self.qubit_array[i][qubit] = measurement
                measured_state = np.argmax(np.diag(self.qubit_array[i].rho))
                self.bits.add_bits(str(measured_state % 2))
                if self.verbose:
                    measurement.set_display_mode("density")
                    print(f"Measured the state {measurement} of qubit {qubit}")
            else:
                measurement = Measure(state=self.qubit_array[i]).measure_state(povm)
                self.qubit_array[i] = measurement
                measured_state = np.argmax(np.diag(self.qubit_array[i].rho))
                self.bits.add_bits(str(measured_state % 2))
                if self.verbose:
                    print(f"Measured the state {measurement} of the whole system")
    


    
        
    def get_info(self, index=0) -> float:
        return QuantInfo.state_info(self.qubit_array[index])
    
    
    def purity(self, index=0, qubit: Qubit=None) -> float:
        """returns the purity of the state or qubit"""
        purity = QuantInfo.purity(self.qubit_array[index][qubit]) if qubit else QuantInfo.purity(self.qubit_array[index][qubit])
        if self.verbose:
            print(f"Purity of the qubit {qubit} is {purity}") if qubit else print(f"Purity of the state is {purity}")
        return purity
    
    def linear_entropy(self, qubit: Qubit=None) -> float:
        """returns the linear entropy of the state or qubit"""
        linear_entropy = QuantInfo.linear_entropy(self.state[qubit]) if qubit else QuantInfo.linear_entropy(self.state)
        if self.verbose:
            print(f"Linear Entropy of the qubit {qubit} is {linear_entropy}") if qubit else print(f"Linear Entropy of the state is {linear_entropy}")
        return linear_entropy
    
    def vn_entropy(self, qubit: Qubit=None) -> float:
        """returns the von neumann entropy of the state or qubit"""
        return QuantInfo.vn_entropy(self.state[qubit]) if qubit else QuantInfo.vn_entropy(self.state)
    
    def shannon_entropy(self, qubit: Qubit=None) -> float:
        """returns the shannon entropy of the state or qubit"""
        return QuantInfo.shannon_entropy(self.state[qubit]) if qubit else QuantInfo.shannon_entropy(self.state)
    
    def single_kraus_generator(self, channel: str, prob: float) -> tuple[Gate, Gate] | tuple[Gate, Gate, Gate, Gate]:
        """Generates the kraus operators for the specific quantum channel"""
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
    
    def apply_channel_to_qubit(self, qubit: int, channel: str, prob: float) -> Qubit:
        """Applies a channel to a specific qubit"""
        if self.collapsed:
            QuantumCircuitError(f"Cannot apply a quantum channel to a collapsed state")
        qubit_state = self.state[qubit]
        old_name = qubit_state.name
        kraus_ops = self.single_kraus_generator(channel, prob)
        kraus_validation(kraus_ops)
        epsilon_rho = np.zeros((2, 2), dtype=np.complex128)
        epsilon = Qubit(rho=epsilon_rho, skip_validation=True)
        qubit_state.skip_val = True
        for k in kraus_ops:
            k_applied = k @ qubit_state
            epsilon += k_applied
        kwargs = {"rho": epsilon.rho, "skip_validation": False, "name": f"{channel} channel applied to {old_name}"}
        qubit_state = Qubit(**kwargs)
        self.state[qubit] = qubit_state
        return self.state

    def apply_local_channel_to_state(self, channel: str, prob: float) -> Qubit:
        """Applies a channel to an entire state"""
        for i in range(self.qubit_num):
            self.apply_channel_to_qubit(i, channel, prob)
        return self.state

    def debug(self, title: bool=True) -> None:
        """Lists some debug information and also calls the debug function in the Qubit class"""
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