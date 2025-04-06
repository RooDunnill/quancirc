import numpy as np
import logging
from scipy import sparse
from .circuit_errors import LwStatePreparationError, LwMeasureError


def lw_qubit_validation(state) -> None:
    logging.debug(f"Starting lightweight Qubit validation")
    if state.state is None:
        raise LwStatePreparationError(f"self.state must be provided to create the lightweight Quantum state")
    if isinstance(state.state, (list, np.ndarray)):
        state.state = np.array(state.state, dtype=np.complex128)
    elif sparse.issparse(state.state):
        state.state = sparse.csr_matrix(state.state, dtype=np.complex128)
    else:
        raise LwStatePreparationError(f"The inputted self.state cannot be of type {type(state.state)}, expected type list or type np.ndarray")
    logging.debug(f"Starting detailed lightweight Qubit validation")
    if not state.skip_val:
        if not isinstance(state.name, str):
            raise LwStatePreparationError(f"The inputted self.name cannot be of type {type(state.name)}, expected type str")
        if not isinstance(state.display_mode, str):
            raise LwStatePreparationError(f"The inputted self.display_mode cannot be of type {type(state.display_mode)}, expected type str")
        logging.debug(f"Ending detailed lightweight Qubit validation")
    logging.debug(f"Ending lightweight Qubit validation")
        
def lw_measure_validation(measure):
    if measure.state.class_type != "lwqubit":
        raise LwMeasureError(f"self.state cannot be of type {type(measure.state)}, expected type LwQubit")