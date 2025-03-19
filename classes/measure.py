import numpy as np
from utilities.qc_errors import MeasureError
from random import choices, randint
from classes.qubit import *


class Measure:
    def __init__(self, state):
        self.class_type = "measure"
        self.state = state

    def list_probs(self, povm: np.ndarray=None) -> np.ndarray:
        if self.state.class_type == "qubit":
            if povm is not None:
                probs = np.array([np.real(np.trace(P * self.density.rho)) for P in povm], dtype=np.float64)
                return probs
            if povm is None:
                root_probs = np.diagonal(self.state.rho)
                probs = (np.conj(root_probs) * root_probs).real
                if np.isclose(np.sum(probs), 1.0, atol=1e-5):
                    return probs
                raise MeasureError(f"The sum of the probabilities adds up to {np.sum(probs)}, however it must sum to 1 to be valid")
        raise MeasureError(f"Inputted state cannot be of type {type(self.state)}, expected Qubit class")
       

    def measure_state(self, povm: np.ndarray = None) -> str:
        probs = self.list_probs(povm)
        measurement = choices(range(len(probs)), weights=probs)[0]
        if povm is not None:
            return measurement
        else:
            post_measurement_vector = np.zeros((self.state.dim,1), dtype=np.complex128)
            post_measurement_vector[measurement] = 1
            post_measurement_density = np.outer(post_measurement_vector, post_measurement_vector.conj())
            return Qubit(rho=post_measurement_density, state_type=self.state.state_type)

        if qubit is None:
            num_bits = int(np.log2(self.state.dim))
            if text:
                print_array(f"Measured the state: |{bin(self.measurement)[2:].zfill(num_bits)}>")
            self.state.vector[:] = 0
            self.state.vector[self.measurement] = 1
            self.state.vector = self.state.vector / np.linalg.norm(self.state.vector)
            return self.measurement, self.state
        elif isinstance(qubit, int):
            if self.measurement == 0:
                measure_rho = np.array([1,0,0,0])
            elif self.measurement == 1:
                measure_rho = np.array([0,0,0,1])
            num_bits = int(np.log2(1))
            if text:
                print_array(f"Measured the {qubit} qubit in state |{bin(self.measurement)[2:].zfill(num_bits)}>")
            post_measurement_den = Density(rho=A_rho) @ Density(rho=measure_rho) @ Density(rho=B_rho)
            pm_state = Qubit(vector=diagonal(post_measurement_den.rho))
            pm_state.norm()
            return self.measurement, pm_state
        else:
            MeasurementError(f"Inputted qubit cannot be of type {type(qubit)}, expected int")





"""
class Measure(StrMixin, LinearMixin):

    array_name = "probs"
    def __init__(self, **kwargs):
        self.measurement_qubit = kwargs.get("m_qubit", "all")
        self.measure_type: str = kwargs.get("type", "projective")
        self.state = kwargs.get("state", None)
        self.name = kwargs.get("name", f"Measurement of state")
        self.fast = kwargs.get("fast", False)
        if not self.fast:
            if self.state is not None:
                self.density: Density = kwargs.get("density", Density(state=self.state))
                self.rho: np.ndarray = self.density.rho
                if self.rho is not None:
                    self.length = len(self.rho)
                    self.dim = int(np.sqrt(self.length))
                    self.n = int(np.log2(self.dim))
            else:
                self.density = kwargs.get("density", None)
                self.rho = self.density.rho if isinstance(self.density, Density) else kwargs.get("rho", None)
                if self.rho is not None:
                    self.length = len(self.rho)
                    self.dim = int(np.sqrt(self.length))
                    self.n = int(np.log2(self.dim))
        self.probs = self.list_probs()
        self.pm_state = None
        self.measurement = None

    def topn_measure_probs(self, qubit: int=None, povm: np.ndarray=None, **kwargs) -> np.ndarray:
    
        topn = kwargs.get("n", 8)
        return top_probs(self.list_probs(qubit, povm), topn)
    
    def list_probs(self, qubit: int=None, povm: np.ndarray=None) -> np.ndarray:

        if povm is not None:
            self.probs = np.array([np.real(trace(P * self.density.rho)) for P in povm], dtype=np.float64)
            return self.probs
        if qubit is None:
            if self.fast:
                vector = self.state.vector
                return np.real(np.multiply(vector, np.conj(vector)))
            elif isinstance(self.density, Density):
                if self.rho is None:
                    self.rho = self.density.rho
                self.probs = np.array([self.rho[i + i * self.density.dim].real for i in range(self.density.dim)], dtype=np.float64)
                return self.probs
            raise MeasurementError(f"Must either be running in fast, or self.density is of the wrong type {type(self.density)}, expected Density class")
        if qubit is not None:
            if qubit > self.n - 1:
                raise QC_error(f"The chosen qubit {qubit}, must be no more than the number of qubits in the circuit {self.n}")
            trace_density = self.density
            if qubit == 0:
                A_rho = np.array([1+1j])
                B_rho = trace_density.partial_trace(trace_out="A", state_size = 1)
                measure_rho = trace_density.partial_trace(trace_out="B", state_size = self.n - 1)
            elif qubit == self.n - 1:
                A_rho = trace_density.partial_trace(trace_out="B", state_size = 1)
                B_rho = np.array([1+1j])
                measure_rho = trace_density.partial_trace(trace_out="A", state_size = self.n - 1)
            elif isinstance(qubit, int):
                A_rho = trace_density.partial_trace(trace_out="B", state_size = self.n - qubit)
                B_rho = trace_density.partial_trace(trace_out="A", state_size = qubit + 1)
                measure_den = Density(rho=A_rho)
                measure_rho = measure_den.partial_trace(trace_out="A", state_size = measure_den.n - 1)
            else:
                raise MeasurementError(f"Inputted qubit cannot be of type {type(qubit)}, expected int") 
            
            measure_den = Density(rho=measure_rho)
            if povm is not None:
                self.probs = np.array([np.real(trace(P * measure_den.rho)) for P in povm], dtype=np.float64)
                return self.probs, measure_rho, A_rho, B_rho
            if povm is None:
                self.probs = np.array([measure_den.rho[i + i * measure_den.dim].real for i in range(measure_den.dim)], dtype=np.float64)
                return self.probs, measure_rho, A_rho, B_rho
            
    def measure_state(self, qubit: int = None, povm: np.ndarray = None, text: bool = False) -> str:

        if qubit is not None:
            probs, measure_rho, A_rho, B_rho = self.list_probs(qubit, povm)
        elif qubit is None:
            probs = self.list_probs(qubit, povm)
        self.measurement = choices(range(len(probs)), weights=probs)[0]
        if povm is not None:
            if text:
                print_array(f"Measured POVM outcome: {povm[self.measurement]}")
            return self.measurement
        
        if qubit is None:
            num_bits = int(np.log2(self.state.dim))
            if text:
                print_array(f"Measured the state: |{bin(self.measurement)[2:].zfill(num_bits)}>")
            self.state.vector[:] = 0
            self.state.vector[self.measurement] = 1
            self.state.vector = self.state.vector / np.linalg.norm(self.state.vector)
            return self.measurement, self.state
        elif isinstance(qubit, int):
            if self.measurement == 0:
                measure_rho = np.array([1,0,0,0])
            elif self.measurement == 1:
                measure_rho = np.array([0,0,0,1])
            num_bits = int(np.log2(1))
            if text:
                print_array(f"Measured the {qubit} qubit in state |{bin(self.measurement)[2:].zfill(num_bits)}>")
            post_measurement_den = Density(rho=A_rho) @ Density(rho=measure_rho) @ Density(rho=B_rho)
            pm_state = Qubit(vector=diagonal(post_measurement_den.rho))
            pm_state.norm()
            return self.measurement, pm_state
        else:
            MeasurementError(f"Inputted qubit cannot be of type {type(qubit)}, expected int") 
            """