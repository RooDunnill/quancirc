import numpy as np
import scipy as sp
from scipy import sparse
from ..circuit_config import sparse_threshold


def convert_to_sparse(matrix):
    if isinstance(matrix, sparse.spmatrix):
        return matrix
    
    zero_fraction = np.count_nonzero(matrix == 0) / len(matrix)**2

    if zero_fraction >= sparse_threshold:
        return sparse.csr_matrix(matrix)
    return matrix

def convert_to_dense(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    
    zero_fraction = np.count_nonzero(matrix == 0) / len(matrix)**2

    if zero_fraction <= sparse_threshold and isinstance(matrix, sparse.spmatrix):
        return matrix.todense()
    
def sparse_mat(matrix):
    if isinstance(matrix, sparse.spmatrix):
        return matrix
    else:
        return sparse.csr_matrix(matrix)

def dense_mat(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    else:
        return matrix.todense()