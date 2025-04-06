import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, norm
import logging
from ....config.config import eig_threshold
from ...circuit_utilities.circuit_errors import *
from ...circuit_utilities.sparse_funcs import *
from .circuit_errors import *


def qutrit_validation(state) -> None:
    """Checks if a density matrix is valid in __init__, returns type None"""
    logging.debug(f"Running Qubit validation of {state.id}")
    if state.state is not None:
        if isinstance(state.state, (list, np.ndarray)):
            state.state = np.array(state.state, dtype=np.complex128)
        elif sparse.issparse(state.state):
            state.state = sparse.csr_matrix(state.state, dtype=np.complex128)
        else:
            raise QutritStatePreparationError(f"The inputted self.state cannot be of type {type(state.state)}, expected type list or type np.ndarray")
        if len(state.state.shape) != 1 and state.weights is None:
            raise QutritStatePreparationError(f"The inputted self.state must be 1D not {state.state.shape}")
    if state.weights is not None:
        if isinstance(state.weights, (list, np.ndarray)):
            state.weights = np.array(state.weights, dtype=np.float64)
        elif sparse.issparse(state.weights):
            state.state = sparse.csr_matrix(state.weights, dtype=np.complex128)
        else:
            raise QutritStatePreparationError(f"The inputted self.weights cannot be of type {type(state.weights)}, expected type list or type np.ndarray")
    if not isinstance(state.skip_val, bool):
        raise QutritStatePreparationError(f"The inputted self.skip_val cannot be of type {type(state.skip_val)}, expected type bool")
    
    if state.skip_val == False:
        logging.debug(f"Starting detailed Qubit validation of {state.id}")
        if state.weights is not None:
            if not np.isrealobj(state.weights):
                raise QutritStatePreparationError(f"self.weights must be made up of real numbers as it is the probabilities of specific states")
            if len(state.state) != len(state.weights):
                raise QutritStatePreparationError(f"The amount of imputted vectors must be the same as the number of inputted weights")
            if not np.isclose(np.sum(state.weights), 1.0, atol=1e-4):
                raise QutritStatePreparationError(f"The sum of the probabilities must equal 1, not {np.sum(state.weights)}")
        if state.state is not None and state.weights is None:
            sum_check = np.dot(state.state , np.conj(state.state))
            if not np.isclose(sum_check, 1.0, atol=1e-4):
                raise QutritStatePreparationError(f"The absolute square of the elements of the state must sum to 1, not to {sum_check}")
        logging.debug(f"Finished detailed Qubit validation of {state.id}")
    logging.debug(f"Finished Qubit validation of {state.id}")



def rho_validation(state):
    logging.debug(f"Starting rho matrix validation")
    if sparse.issparse(state.rho):
        state.rho = sparse.csr_matrix(state.rho, dtype=np.complex128)
    elif isinstance(state.rho, (list, np.ndarray)):
        state.rho = dense_mat(state.rho)
    else:
        raise QutritStatePreparationError(f"The inputted self.rho cannot be of type {type(state.rho)}, expected type list or type np.ndarray")
    if state.skip_val == False:
        logging.debug(f"Starting detailed rho matrix validation")
        if sparse.issparse(state.rho):
            diff = state.rho - state.rho.getH()
            if norm(diff, "fro") > 1e-4:  
                raise QutritStatePreparationError(f"Density matrix is not Hermitian")
            trace_sparse = state.rho.diagonal().sum()
            if not np.isclose(trace_sparse, 1.0):
                raise QutritStatePreparationError(f"Density matrix must have a trace of 1, not {trace_sparse}")
        else:
            if not np.allclose(state.rho, (state.rho.conj()).T, atol=1e-4):  
                raise QutritStatePreparationError(f"Density matrix is not Hermitian: {state.rho}")
            if not np.isclose(np.trace(state.rho), 1.0):
                raise QutritStatePreparationError(f"Density matrix must have a trace of 1, not of trace {np.trace(state.rho)}")
        if state.rho.shape != (1,1):
            k = 1
            if state.rho.shape[0] < eig_threshold:
                eigenvalues = np.linalg.eigvalsh(dense_mat(state.rho))
            else:
                eigenvalues = eigsh(state.rho, k=k, which="SA", return_eigenvectors=False) if sparse.issparse(state.rho) else np.linalg.eigvalsh(state.rho)
            if np.any(eigenvalues < -1e-4):
                negative_indices = np.where(eigenvalues < 0)[0]
                raise QutritStatePreparationError(f"Density matrix is not positive semi-definite. "
                                    f"Negative eigenvalues found at indices {negative_indices}")
        logging.debug(f"Ending rho matrix validation")