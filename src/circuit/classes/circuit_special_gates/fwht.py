import numpy as np
from ..qubit import *
from ...circuit_utilities.circuit_errors import *

def vector_fwht(state: Qubit) -> Qubit:
        """The Fast Walsh Hadamard Transform, used heavily in Grover's to apply the tensored Hadamard"""
        if isinstance(state, Qubit):
            sqrt2_inv = 1/np.sqrt(2)
            vec = state.state
            for i in range(state.n):                                            #loops through each size of qubit below the size of the state
                step_size = 2**(i + 1)                                          #is the dim of the current qubit tieration size 
                half_step = step_size // 2                                      #half the step size to go between odd indexes
                outer_range = np.arange(0, state.dim, step_size)[:, None]       #more efficient intergration of a loop over the state dim in steps of the current dim 
                inner_range = np.arange(half_step)                               
                indices = outer_range + inner_range                        
                a, b = vec[indices], vec[indices + half_step]
                vec[indices] = (a + b) * sqrt2_inv
                vec[indices + half_step] = (a - b) * sqrt2_inv                        #normalisation has been taken out giving a slight speed up in performance
            return state
        raise TypeError(f"This can't act on this type, only on Qubits")