import numpy as np
from scipy import sparse
from ..utilities.circuit_errors import LwMeasureError
from random import choices, randint
from .lw_qubit import *
from ..utilities.validation_funcs import lw_measure_validation
from ...circuit_utilities.sparse_funcs import dense_mat
from ...circuit_utilities.layout_funcs import *
from ...general_circuit.classes.qubit import copy_qubit_attr

__all__ = ["LwMeasure"]


class LwMeasure:
    def __init__(self, state, **kwargs):
        object.__setattr__(self, 'class_type', 'lwmeasure')
        self.state = state
        lw_measure_validation(self)
        
    def __dir__(cls):
        return ["list_probs", "measure_state"]

    def list_probs(self, qubit=None):
        if qubit is None:
            if sparse.issparse(self.state.state):
                self.prob_distribution = self.state.state.multiply(self.state.state.conjugate())
            else:
                self.prob_distribution = np.einsum("i,i->i", self.state.state.ravel(), np.conj(self.state.state.ravel()))
            self.prob_distribution = dense_mat(self.prob_distribution).ravel().real
        return self.prob_distribution

    def measure_state(self, qubit=None):
        if qubit is None:
            probs = self.list_probs()
            measurement = choices(range(len(probs)), weights=probs)[0]
            post_measurement_vector = np.zeros((self.state.dim), dtype=np.complex128)
            post_measurement_vector[measurement] = 1
            kwargs = {"state": post_measurement_vector}
            kwargs.update(copy_qubit_attr(self))
            self.collapsed = True
            return LwQubit(**kwargs)

    