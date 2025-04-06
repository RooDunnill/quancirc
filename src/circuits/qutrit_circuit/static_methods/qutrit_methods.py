import numpy as np
from ..utilities.gen_utilities import *


def q0_state(cls, **kwargs):
    """The |0> Qubit"""
    n = kwargs.get("n", 1)
    q0_vector = [1,0,0]
    q0_id = f"|0>"
    if n != 1:
        q0_vector = np.zeros(3**n, dtype=np.complex128)
        q0_vector[0] = 1
        zeros_str = "0" * n
        q0_id = f"|{zeros_str}>"
    return cls(id=q0_id, state=q0_vector)

def q1_state(cls, **kwargs):
    """The |1> Qubit"""
    n = kwargs.get("n", 1)
    q1_vector = [0,1,0]
    q1_id = f"|1>"
    if n != 1:
        q1_vector = np.zeros(3**n, dtype=np.complex128)
        q1_vector[np.floor(3**n)] = 1
        ones_str = "1" * n
        q1_id = f"|{ones_str}>"
    return cls(id=q1_id, state=q1_vector)

def q2_state(cls, **kwargs):
    """The |2> Qubit"""
    n = kwargs.get("n", 1)
    q2_vector = [0,0,1]
    q2_id = f"|2>"
    if n != 1:
        q2_vector = np.zeros(3**n, dtype=np.complex128)
        q2_vector[-1] = 1
        twos_str = "2" * n
        q2_id = f"|{twos_str}>"
    return cls(id=q2_id, state=q2_vector)