import numpy as np
from utilities.qc_errors import MeasureError
from random import choices, randint
from classes.qubit import *
from utilities.validation_funcs import measure_validation


            


class Measure:
    def __init__(self, state, **kwargs):
        self.class_type = "measure"
        self.state = state
        measure_validation(self)
        
        
   

    def list_probs(self, povm: np.ndarray=None) -> np.ndarray:
        if isinstance(self.state, Qubit):
            if povm:
                if isinstance(povm, (np.ndarray, list)):
                    probs = np.array([np.real(np.trace(P @ self.state.rho)) for P in povm], dtype=np.float64)
                    return probs
                raise MeasureError(f"Inputted povm cannot be of type {type(povm)}, expected np.ndarray")
            if not povm:
                if isinstance(self.state.rho, np.ndarray):
                    probs = np.diagonal(self.state.rho).real
                    if np.isclose(np.sum(probs), 1.0, atol=1e-5):
                        return probs
                    raise MeasureError(f"The sum of the probabilities adds up to {np.sum(probs)}, however it must sum to 1 to be valid, probs: {probs}")
                raise MeasureError(f"Inputted state cannot have a state.rho of type  {type(self.state.rho)}, expected np.ndarray")
        raise MeasureError(f"Inputted state cannot be of type {type(self.state)}, expected Qubit class")
       
    def measure_state(self, povm: np.ndarray = None) -> Qubit:
        probs = self.list_probs(povm)
        measurement = choices(range(len(probs)), weights=probs)[0]
        if povm:
                measurement_povm = povm[measurement]
                post_measurement_density = np.dot(np.dot(measurement_povm, self.state.rho), np.conj(measurement_povm))
                norm = np.trace(post_measurement_density)
                return Qubit(rho=(post_measurement_density / norm), state_type=self.state.state_type)
        else:
            post_measurement_vector = np.zeros((self.state.dim,1), dtype=np.complex128)
            post_measurement_vector[measurement] = 1
            post_measurement_density = np.outer(post_measurement_vector, post_measurement_vector.conj())
            return Qubit(rho=post_measurement_density, state_type=self.state.state_type)

    