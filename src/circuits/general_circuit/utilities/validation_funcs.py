import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, norm
import logging
from ....config.config import eig_threshold
from ...circuit_utilities.circuit_errors import *
from ...circuit_utilities.sparse_funcs import *
from .circuit_errors import *

def qubit_validation(state) -> None:
    """Checks if a density matrix is valid in __init__, returns type None"""
    logging.debug(f"Running Qubit validation of {state.id}")
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
    
    if state.skip_val == False:
        logging.debug(f"Starting detailed Qubit validation of {state.id}")
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
        logging.debug(f"Finished detailed Qubit validation of {state.id}")
    logging.debug(f"Finished Qubit validation of {state.id}")
            
def qubit_array_validation(array) -> None:
    logging.debug(f"Starting QubitArray validation of {array.name}")
    if not isinstance(array.name, str):
        raise QubitArrayError(f"self.name cannot be of type {type(array.name)}, expected type str")
    if not isinstance(array.qubit_array, list):
        raise QubitArrayError(f"self.qubit_array cannot be of type {type(array.qubit_array)}, expected type list")
    if len(array) > 0:
        if not all(type(i) == type(array.qubit_array[0]) for i in array.qubit_array) or not array.qubit_array[0].class_type == "qubit":
            raise QubitArrayError(f"Every element in the array must be of type Qubit")
    else:
        raise QubitArrayError(f"There must be atleast one qubit in the arry, not {len(array)} qubits")
    logging.debug(f"Ending QubitArray validation of {array.name}")
    
def rho_validation(state):
    logging.debug(f"Starting rho matrix validation")
    if sparse.issparse(state.rho):
        state.rho = sparse.csr_matrix(state.rho, dtype=np.complex128)
    elif isinstance(state.rho, (list, np.ndarray)):
        state.rho = dense_mat(state.rho)
    else:
        raise StatePreparationError(f"The inputted self.rho cannot be of type {type(state.rho)}, expected type list or type np.ndarray")
    if state.skip_val == False:
        logging.debug(f"Starting detailed rho matrix validation")
        if sparse.issparse(state.rho):
            diff = state.rho - state.rho.getH()
            if norm(diff, "fro") > 1e-4:  
                raise StatePreparationError(f"Density matrix is not Hermitian")
            trace_sparse = state.rho.diagonal().sum()
            if not np.isclose(trace_sparse, 1.0):
                raise StatePreparationError(f"Density matrix must have a trace of 1, not {trace_sparse}")
        else:
            if not np.allclose(state.rho, (state.rho.conj()).T, atol=1e-4):  
                raise StatePreparationError(f"Density matrix is not Hermitian: {state.rho}")
            if not np.isclose(np.trace(state.rho), 1.0):
                raise StatePreparationError(f"Density matrix must have a trace of 1, not of trace {np.trace(state.rho)}")
        if state.rho.shape != (1,1):
            k = 1
            if state.rho.shape[0] < eig_threshold:
                eigenvalues = np.linalg.eigvalsh(dense_mat(state.rho))
            else:
                eigenvalues = eigsh(state.rho, k=k, which="SA", return_eigenvectors=False) if sparse.issparse(state.rho) else np.linalg.eigvalsh(state.rho)
            if np.any(eigenvalues < -1e-4):
                negative_indices = np.where(eigenvalues < 0)[0]
                raise StatePreparationError(f"Density matrix is not positive semi-definite. "
                                    f"Negative eigenvalues found at indices {negative_indices}")
        logging.debug(f"Ending rho matrix validation")
            

def gate_validation(gate) -> None:
    logging.debug(f"Starting Gate validation of {gate.name}")
    if not isinstance(gate.name, str):
        raise GateError(f"self.name cannot be of type: {type(gate.name)}, expected type str")
    if gate.matrix is None:
        raise GateError(f"Gates can only be initialised if they are provided with a matrix")
    if sparse.issparse(gate.matrix) or isinstance(gate.matrix[0], sparse.spmatrix):
        gate.matrix = sparse.csr_matrix(gate.matrix, dtype=np.complex128)
        logging.debug(f"Redefining Gate matrix as a sparse csr matrix")
    elif isinstance(gate.matrix, (list, np.ndarray)):
        gate.matrix = np.array(gate.matrix, dtype=np.complex128)
        logging.debug(f"Redefining Gate matrix as a numpy array")
    else:
        raise GateError(f"The gate cannot be of type: {type(gate.matrix)}, expected type list or np.ndarray")
    
    if gate.skip_val == False:
        logging.debug(f"Starting detailed Gate validation of {gate.name}")
        if np.size(gate.matrix) != 1:
            if gate.matrix.shape[0] != gate.matrix.shape[1]:
                raise GateError(f"All gates must be of a square shape. This gate has shape {gate.matrix.shape[0]} x {gate.matrix.shape[1]}")
        if sparse.issparse(gate.matrix):
            gate_adjoint = gate.matrix.T
            gate_check = gate_adjoint.dot(gate.matrix)
            diag_elements = gate_check.diagonal()  
            if not np.all(np.isclose(np.abs(diag_elements), 1.0, atol=1e-4)):
                raise GateError(f"This gate is not unitary {gate.matrix}")
        else:
            gate_check = np.dot(np.conj(gate.matrix.T), gate.matrix)
            if not np.all(np.isclose(np.diag(gate_check),1.0, atol=1e-4)):
                logging.critical(f"Gate Check: {gate_check}")
                raise GateError(f"This gate is not unitary {gate.matrix}")
        logging.debug(f"Ending detailed Gate validation of {gate.name}")
    logging.debug(f"Ending Gate validation of {gate.name}")
        
def measure_validation(measure):
    logging.debug(f"Starting Measure validation")
    if measure.state.class_type != "qubit":
        raise MeasureError(f"self.state cannot be of type {type(measure.state)}, expected type Qubit")
    logging.debug(f"Ending Measure validation")
    
def circuit_validation(circuit):
    logging.debug(f"Starting Circuit validation of {circuit.name}")
    if not isinstance(circuit.qubit_num, int):
        raise QuantumCircuitError(f"self.qubit_num cannot be of type {type(circuit.qubit_num)}, expected type int")
    if not isinstance(circuit.bit_num, int):
        raise QuantumCircuitError(f"self.bit_num cannot be of type {type(circuit.bit_num)}, expected type int")
    if not isinstance(circuit.verbose, bool):
        raise QuantumCircuitError(f"self.verbose cannot be of type {type(circuit.bit_num)}, expected type bool")
    logging.debug(f"Ending Circuit validation of {circuit.name}")

def kraus_validation(kraus_operators: list | tuple | np.ndarray):
    logging.debug(f"Starting Kraus operators validation")
    if isinstance(kraus_operators, (list, tuple, np.ndarray)):
        kraus_dim = len(kraus_operators[0].matrix)
        kraus_sum = np.zeros((kraus_dim, kraus_dim), dtype=np.complex128)
        for i in kraus_operators:
            kraus = i.matrix
            kraus_sum += np.conj(kraus).T @ kraus
        if not np.allclose(kraus_sum, np.eye(kraus_dim)):
            raise QuantumCircuitError(f"These kraus operators are invalid as they don't sum to the identity matrix.")
    logging.debug(f"Ending Kraus operators validation")