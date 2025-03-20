import numpy as np
from utilities.qc_errors import *

def qubit_validation(state) -> None:
    """Checks if a density matrix is valid in __init__, returns type None"""
    if not state.skip_val:
        pass

        
            
def rho_validation(state):
    if not state.skip_val:
        if not isinstance(state.rho, (list, np.ndarray)):
            raise StatePreparationError(f"The inputted self.rho cannot be of type {type(state.rho)}, expected list or np.ndarray")
        state.rho = np.array(state.rho, dtype=np.complex128)
        if not np.allclose(state.rho, state.rho.conj().T):  
            raise StatePreparationError(f"Density matrix is not Hermitian: {state.rho}")
        if not np.array_equal(state.rho, np.array([1])):
            eigenvalues = np.linalg.eigvalsh(state.rho)
            if np.any(eigenvalues < -1e-4):
                negative_indices = np.where(eigenvalues < 0)[0]
                raise StatePreparationError(f"Density matrix is not positive semi-definite. "
                                    f"Negative eigenvalues found at indices {negative_indices}")
            if not np.isclose(np.trace(state.rho), 1.0):
                raise StatePreparationError(f"Density matrix must have a trace of 1, not of trace {np.trace(state.rho)}")

def gate_validation(gate):
    if not isinstance(gate.name, str):
        raise GateError(f"self.name cannot be of type: {type(gate.name)}, expected type str")
    if gate.matrix is None:
        raise GateError(f"Gates can only be initialised if they are provided with a matrix")
    if not isinstance(gate.matrix, (list, np.ndarray)):
        raise GateError(f"The gate cannot be of type: {type(gate.matrix)}, expected type list or np.ndarray")
    gate.matrix = np.array(gate.matrix, dtype=np.complex128)
    if np.size(gate.matrix) != 1:
        if gate.matrix.shape[0] != gate.matrix.shape[1]:
            raise GateError(f"All gates must be of a square shape. This gate has shape {gate.matrix.shape[0]} x {gate.matrix.shape[1]}")
        gate_check = np.dot(np.conj(gate.matrix.T), gate.matrix)
        if not np.all(np.isclose(np.diag(gate_check),1.0, atol=1e-3)):
            raise GateError(f"This gate is not unitary {gate.matrix}")
        
def measure_validation(measure):
    if measure.state.class_type != "qubit":
        raise MeasureError(f"self.state cannot be of type {type(measure.state)}, expected type Qubit")
    
def circuit_validation(circuit):
    if not isinstance(circuit.qubit_num, int):
        raise QuantumCircuitError(f"self.qubit_num cannot be of type {type(circuit.qubit_num)}, expected type int")
    if not isinstance(circuit.bit_num, int):
        raise QuantumCircuitError(f"self.bit_num cannot be of type {type(circuit.bit_num)}, expected type int")
    if not isinstance(circuit.verbose, bool):
        raise QuantumCircuitError(f"self.verbose cannot be of type {type(circuit.bit_num)}, expected type bool")
