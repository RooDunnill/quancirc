import numpy as np
from ..circuit_utilities import SymbQuantInfoError
from .symb_qubit import *
from scipy.linalg import sqrtm, logm
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from ..circuit_config import *
import sympy as sp
from scipy.linalg import sqrtm

__all__ = ["SymbQuantInfo"]


class SymbQuantInfo:

    @staticmethod
    def trace_distance(state_1, state_2):
        delta_state = state_1 - state_2
        eigenvalues = delta_state.rho.eigenvals()
        abs_eigenvalues = sum(sp.Abs(lam) for lam in eigenvalues)
        trace_distance = 0.5 * abs_eigenvalues
        return trace_distance.simplify()
    
    @staticmethod
    def trace_distance_bound(state_1:SymbQubit, state_2: SymbQubit) -> tuple[float, float]:
        fidelity = SymbQuantInfo.fidelity(state_1, state_2)
        lower = 1 - fidelity**0.5
        upper = (1 - fidelity)**0.5
        return lower, upper
    
    @staticmethod
    def fidelity(state_1: SymbQubit, state_2: SymbQubit) -> float:
        if state_1 is None or state_2 is None:
            raise SymbQuantInfoError(f"Must provide two states of type Qubit")
        if isinstance(state_1, SymbQubit) and isinstance(state_2, SymbQubit):
            rho_1 = state_1.rho
            rho_2 = state_2.rho
            product = rho_1**0.5 @ rho_2 @ rho_1**0.5
            sqrt_product = product**0.5
            mat_trace = sqrt_product.trace()
            fidelity = sp.re(mat_trace*sp.conjugate(mat_trace))
            return fidelity
        raise SymbQuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")