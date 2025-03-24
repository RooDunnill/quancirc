import numpy as np
import scipy as sp
from scipy import sparse
from ..circuit_config import sparse_threshold
from ..circuit_utilities.circuit_errors import SparseMatrixError

def count_zeros(matrix):
    if isinstance(matrix, np.ndarray):
        return np.count_nonzero(matrix == 0)
    elif sparse.issparse(matrix):
        return matrix.size - matrix.getnnz()
    else:
        raise ValueError("Input must be a numpy array or a sparse matrix.")

def convert_to_sparse(matrix):
    if sparse.issparse(matrix):
        return matrix
    zero_fraction = count_zeros(matrix) / matrix.size
    if zero_fraction >= sparse_threshold:
        return sparse.csr_matrix(matrix)
    return matrix

def convert_to_dense(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    zero_fraction = count_zeros(matrix) / matrix.size
    if zero_fraction <= sparse_threshold and sparse.issparse(matrix):
        return np.asarray(matrix.todense(), dtype=np.complex128)
    
def sparse_mat(matrix):
    if sparse.issparse(matrix):
        return matrix
    else:
        return sparse.csr_matrix(matrix)

def dense_mat(matrix):
    if isinstance(matrix, sparse.spmatrix): 
        return np.asarray(matrix.todense(), dtype=np.complex128)
    elif isinstance(matrix, np.ndarray):
        return np.asarray(matrix, dtype=np.complex128)
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(matrix)}")