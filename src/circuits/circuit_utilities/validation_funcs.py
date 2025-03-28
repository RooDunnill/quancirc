import numpy as np
from scipy import sparse
from .circuit_errors import *
from .sparse_funcs import *

def qubit_validation(state) -> None:
    """Checks if a density matrix is valid in __init__, returns type None"""
    if state.state is not None:
        if isinstance(state.state, (list, np.ndarray)):
            state.state = np.array(state.state, dtype=np.complex128)
        elif sparse.issparse(state.state):
            state.state = sparse.csr_matrix(state.state, dtype=np.complex128)
        else:
            raise StatePreparationError(f"The inputted self.state cannot be of type {type(state.state)}, expected type list or type np.ndarray")
        
        if len(state.state.shape) != 1 and state.weights is None:
            raise StatePreparationError(f"The inputted self.state must be 1D not {state.state.shape}")
    if state.weights is not None:
        if isinstance(state.weights, (list, np.ndarray)):
            state.weights = np.array(state.weights, dtype=np.float64)
        elif sparse.issparse(state.weights):
            state.state = sparse.csr_matrix(state.weights, dtype=np.complex128)
        else:
            raise StatePreparationError(f"The inputted self.weights cannot be of type {type(state.weights)}, expected type list or type np.ndarray")
    if not isinstance(state.skip_val, bool):
        raise StatePreparationError(f"The inputted self.skip_val cannot be of type {type(state.skip_val)}, expected type bool")
    if not state.skip_val:
        if not isinstance(state.name, str):
            raise StatePreparationError(f"The inputted self.name cannot be of type {type(state.name)}, expected type str")
        if not isinstance(state.display_mode, str):
            raise StatePreparationError(f"The inputted self.display_mode cannot be of type {type(state.display_mode)}, expected type str")
        if state.weights is not None:
            if not np.isrealobj(state.weights):
                raise StatePreparationError(f"self.weights must be made up of real numbers as it is the probabilities of specific states")
            if len(state.state) != len(state.weights):
                raise StatePreparationError(f"The amount of imputted vectors must be the same as the number of inputted weights")
            if not np.isclose(np.sum(state.weights), 1.0, atol=1e-4):
                raise StatePreparationError(f"The sum of the probabilities must equal 1, not {np.sum(state.weights)}")
        if state.state is not None and state.weights is None:
            sum_check = np.dot(state.state , np.conj(state.state))
            if not np.isclose(sum_check, 1.0, atol=1e-4):
                raise StatePreparationError(f"The absolute square of the elements of the state must sum to 1, not to {sum_check}")
            
def lw_qubit_validation(state) -> None:
    if state.state is None:
        raise LWStatePreparationError(f"self.state must be provided to create the lightweight Quantum state")
    if isinstance(state.state, (list, np.ndarray)):
        state.state = np.array(state.state, dtype=np.complex128)
    elif sparse.issparse(state.state):
        state.state = sparse.csr_matrix(state.state, dtype=np.complex128)
    else:
        raise StatePreparationError(f"The inputted self.state cannot be of type {type(state.state)}, expected type list or type np.ndarray")
    if not state.skip_val:
        if not isinstance(state.name, str):
            raise StatePreparationError(f"The inputted self.name cannot be of type {type(state.name)}, expected type str")
        if not isinstance(state.display_mode, str):
            raise StatePreparationError(f"The inputted self.display_mode cannot be of type {type(state.display_mode)}, expected type str")
        
def qubit_array_validation(array) -> None:
    if not isinstance(array.name, str):
        raise QubitArrayError(f"self.name cannot be of type {type(array.name)}, expected type str")
    if not isinstance(array.qubit_array, list):
        raise QubitArrayError(f"self.qubit_array cannot be of type {type(array.qubit_array)}, expected type list")
    if len(array) > 0:
        if not all(type(i) == type(array.qubit_array[0]) for i in array.qubit_array) or not array.qubit_array[0].class_type == "qubit":
            raise QubitArrayError(f"Every element in the array must be of type Qubit")
    else:
        raise QubitArrayError(f"There must be atleast one qubit in the arry, not {len(array)} qubits")
    
def rho_validation(state):
    if sparse.issparse(state.rho) or isinstance(state.rho[0], sparse.spmatrix):
        state.rho = sparse.csr_matrix(state.rho, dtype=np.complex128)
    elif isinstance(state.rho, (list, np.ndarray)):
        state.rho = np.array(dense_mat(state.rho), dtype=np.complex128)
    else:
        raise StatePreparationError(f"The inputted self.rho cannot be of type {type(state.rho)}, expected type list or type np.ndarray")
        
    if not state.skip_val:
        test_rho = dense_mat(state.rho)
        if not np.allclose(test_rho, (test_rho.conj()).T):  
            raise StatePreparationError(f"Density matrix is not Hermitian: {test_rho}")
        if not np.array_equal(test_rho, np.array([1])):
            eigenvalues = np.linalg.eigvalsh(test_rho)
            if np.any(eigenvalues < -1e-4):
                negative_indices = np.where(eigenvalues < 0)[0]
                raise StatePreparationError(f"Density matrix is not positive semi-definite. "
                                    f"Negative eigenvalues found at indices {negative_indices}")
            if not np.isclose(np.trace(test_rho), 1.0):
                raise StatePreparationError(f"Density matrix must have a trace of 1, not of trace {np.trace(test_rho)}")

def gate_validation(gate):
    if not isinstance(gate.name, str):
            raise GateError(f"self.name cannot be of type: {type(gate.name)}, expected type str")
    if gate.matrix is None:
        raise GateError(f"Gates can only be initialised if they are provided with a matrix")
    if sparse.issparse(gate.matrix) or isinstance(gate.matrix[0], sparse.spmatrix):
        gate.matrix = sparse.csr_matrix(gate.matrix, dtype=np.complex128)
    elif isinstance(gate.matrix, (list, np.ndarray)):
        gate.matrix = np.array(gate.matrix, dtype=np.complex128)
    else:
        raise GateError(f"The gate cannot be of type: {type(gate.matrix)}, expected type list or np.ndarray")
    
    if not gate.skip_val:
        if np.size(gate.matrix) != 1:
            if gate.matrix.shape[0] != gate.matrix.shape[1]:
                raise GateError(f"All gates must be of a square shape. This gate has shape {gate.matrix.shape[0]} x {gate.matrix.shape[1]}")
        if sparse.issparse(gate.matrix):
            gate_adjoint = gate.matrix.T
            gate_check = gate_adjoint.dot(gate.matrix)
            diag_elements = gate_check.diagonal()  
            if not np.all(np.isclose(diag_elements, 1.0, atol=1e-3)):
                raise GateError(f"This gate is not unitary {gate.matrix}")
        else:
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

def kraus_validation(kraus_operators: list | tuple | np.ndarray):
    if isinstance(kraus_operators, (list, tuple, np.ndarray)):
        kraus_dim = len(kraus_operators[0].matrix)
        kraus_sum = np.zeros((kraus_dim, kraus_dim), dtype=np.complex128)
        for i in kraus_operators:
            kraus = i.matrix
            kraus_sum += np.conj(kraus).T @ kraus
        if not np.allclose(kraus_sum, np.eye(kraus_dim)):
            raise QuantumCircuitError(f"These kraus operators are invalid as they don't sum to the identity matrix.")