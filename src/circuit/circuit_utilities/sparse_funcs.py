import numpy as np
import scipy as sp
from scipy import sparse
from ..circuit_config import sparse_matrix_threshold, sparse_array_threshold
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
    if zero_fraction >= sparse_matrix_threshold:
        return sparse.csr_matrix(matrix)
    return matrix

def convert_to_sparse_array(array):
    if isinstance(array, sparse.csr_matrix):
        return array
    if sparse.issparse(array):
        return array
    zero_fraction = count_zeros(array) / array.size
    if zero_fraction >= sparse_matrix_threshold:
        return sparse.csr_array(array)
    return array

def convert_to_dense(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    zero_fraction = count_zeros(matrix) / matrix.size
    if zero_fraction <= sparse_matrix_threshold and sparse.issparse(matrix):
        return np.asarray(matrix.todense(), dtype=np.complex128)
    
def convert_to_dense_array(array):
    if isinstance(array, np.ndarray):
        return array.ravel()
    zero_fraction = count_zeros(array) / array.size
    if zero_fraction <= sparse_matrix_threshold and sparse.issparse(array):
        return np.asarray(array.todense(), dtype=np.complex128)
    
def sparse_mat(matrix):
    if sparse.issparse(matrix):
        return matrix
    else:
        return sparse.csr_matrix(matrix)
    
def sparse_array(array):
    if sparse.issparse(array):
        return array
    else:
        return sparse.csr_array(array)

def dense_mat(matrix):
    if isinstance(matrix, sparse.spmatrix): 
        return np.asarray(matrix.todense(), dtype=np.complex128)
    elif isinstance(matrix, np.ndarray):
        return np.asarray(matrix, dtype=np.complex128)
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(matrix)}")

def dense_array(array):
    if isinstance(array, sparse.spmatrix): 
        return np.asarray(array.todense(), dtype=np.complex128)
    elif isinstance(array, np.ndarray):
        return np.asarray(array, dtype=np.complex128).ravel()
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(array)}")