import numpy as np
from scipy import sparse
from ..utilities.circuit_errors import MeasureError
from random import choices, randint
from .qubit import *
from ..utilities.validation_funcs import measure_validation
from ...circuit_utilities.sparse_funcs import dense_mat, sparse_mat


__all__ = ["Measure"]


class Measure:
    def __init__(self, state, **kwargs):
        object.__setattr__(self, 'class_type', 'measure')
        self.state = state
        measure_validation(self)
        
    def __dir__(cls):
        return ["list_probs", "measure_state"]

    def list_probs(self, povm: np.ndarray=None) -> np.ndarray:
        rho = dense_mat(self.state.rho)
        if isinstance(self.state, Qubit):
            if povm:
                if sparse.issparse(povm):
                    probs = np.array([np.real(np.trace(P.dot(rho))) for P in povm], dtype=np.float64)
                    return probs
                elif isinstance(povm, (np.ndarray, list)):
                    probs = np.array([np.real(np.trace(P @ rho)) for P in povm], dtype=np.float64)
                    return probs
                raise MeasureError(f"Inputted povm cannot be of type {type(povm)}, expected np.ndarray")
            if not povm:
                if sparse.issparse(rho):
                    probs = sparse.diags(rho).real
                    probs = dense_mat(probs)
                    if np.isclose(np.sum(probs), 1.0, atol=1e-5):
                        return probs
                    raise MeasureError(f"The sum of the probabilities adds up to {np.sum(probs)}, however it must sum to 1 to be valid, probs: {probs}")
                elif isinstance(rho, np.ndarray):
                    probs = np.diagonal(rho).real
                    if np.isclose(np.sum(probs), 1.0, atol=1e-5):
                        return probs
                    raise MeasureError(f"The sum of the probabilities adds up to {np.sum(probs)}, however it must sum to 1 to be valid, probs: {probs}")
                raise MeasureError(f"Inputted state cannot have a state.rho of type  {type(rho)}, expected np.ndarray")
        raise MeasureError(f"Inputted state cannot be of type {type(self.state)}, expected Qubit class")
       
    def measure_state(self, povm: np.ndarray = None) -> Qubit:         #NEEDS TO CARRY OVER ATTRIBUTES TODOOOOOOOOOOOOOOOOOOOOO
        probs = self.list_probs(povm)
        measurement = choices(range(len(probs)), weights=probs)[0]
        if povm:
            measurement_povm = povm[measurement]
            if sparse.issparse(self.state.rho):
                measurement_povm = sparse_mat(measurement_povm)
                post_measurement_density = measurement_povm @ self.state.rho @ measurement_povm.conj().T
                norm = post_measurement_density.diagonal().sum()
            else:
                post_measurement_density = np.dot(np.dot(measurement_povm, self.state.rho), np.conj(measurement_povm).T)
                norm = np.trace(post_measurement_density)
            return Qubit(rho=(post_measurement_density / norm), state_type=self.state.state_type)
        else:
            post_measurement_vector = np.zeros((self.state.dim,1), dtype=np.complex128)
            post_measurement_vector[measurement] = 1
            post_measurement_density = np.outer(post_measurement_vector, post_measurement_vector.conj())
            return Qubit(rho=post_measurement_density, state_type=self.state.state_type)

    