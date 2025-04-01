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
        return sparse.csr_matrix(matrix, dtype=np.complex128) 
    return matrix

def convert_both_to_sparse(matrix_1, matrix_2):
    zero_fraction_1 = count_zeros(matrix_1) / matrix_1.size
    zero_fraction_2 = count_zeros(matrix_2) / matrix_2.size
    if sparse.issparse(matrix_1) and sparse.issparse(matrix_2):
        return matrix_1, matrix_2
    
    if sparse.issparse(matrix_1):
        if zero_fraction_2 >= sparse_matrix_threshold:
            return sparse.csr_matrix(matrix_1, dtype=np.complex128), sparse.csr_matrix(matrix_2, dtype=np.complex128) 
    if sparse.issparse(matrix_2):
        if zero_fraction_1 >= sparse_matrix_threshold:
            return sparse.csr_matrix(matrix_1, dtype=np.complex128), sparse.csr_matrix(matrix_2, dtype=np.complex128)
    zero_fraction = (count_zeros(matrix_1) + count_zeros(matrix_2))/(matrix_1.size + matrix_2.size)
    if zero_fraction >= sparse_matrix_threshold:
        return sparse.csr_matrix(matrix_1, dtype=np.complex128), sparse.csr_matrix(matrix_2, dtype=np.complex128) 
    return dense_mat(matrix_1), dense_mat(matrix_2)

def convert_to_sparse_array(array):
    if isinstance(array, sparse.csr_matrix):
        return array
    if sparse.issparse(array):
        return array
    zero_fraction = count_zeros(array) / array.size
    if zero_fraction >= sparse_matrix_threshold:
        return sparse.csr_array(array, dtype=np.complex128)
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
        return sparse.csr_matrix(matrix, dtype=np.complex128)
    
def sparse_array(array):
    if sparse.issparse(array):
        return array
    else:
        return sparse.csr_array(array, dtype=np.complex128)

def dense_mat(matrix):
    if isinstance(matrix, sparse.spmatrix): 
        return np.asarray(matrix.todense(), dtype=np.complex128)
    elif isinstance(matrix, np.ndarray):
        return np.asarray(matrix, dtype=np.complex128)
    elif isinstance(matrix, list):
        return np.array(matrix, dtype=np.complex128)
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(matrix)}")

def dense_array(array):
    if isinstance(array, sparse.spmatrix): 
        return np.asarray(array.todense(), dtype=np.complex128)
    elif isinstance(array, np.ndarray):
        return np.asarray(array, dtype=np.complex128).ravel()
    elif isinstance(array, list):
        return np.array(array, dtype=np.complex128).ravel()
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(array)}")