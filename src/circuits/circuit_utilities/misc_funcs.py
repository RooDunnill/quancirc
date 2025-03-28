import numpy as np
from .circuit_errors import QC_error

def binary_entropy(prob: float) -> float:
    """Used to calculate the binary entropy of two probabilities"""
    if isinstance(prob, (float, int)):
        if prob ==  0 or prob == 1:
            return 0.0
        else:
            return -prob*np.log2(prob) - (1 - prob)*np.log2(1 - prob)
    raise QC_error(f"Binary value must be a float")