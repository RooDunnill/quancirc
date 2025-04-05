import numpy as np
import logging
from scipy import sparse
from sympy.matrices.dense import MutableDenseMatrix
from ..circuit_config import *
from ..circuit_utilities.circuit_errors import SparseMatrixError

def count_zeros(matrix):
    if isinstance(matrix, np.ndarray):
        return np.count_nonzero(matrix == 0)
    elif sparse.issparse(matrix):
        mat_size = matrix.shape[0] * matrix.shape[1]
        return mat_size - matrix.getnnz()
    else:
        raise SparseMatrixError(f"Matrix cannot be of type {type(matrix)}, expected type np.ndarray or sparse matrix")
    

@log_function_call
def auto_choose(*matrices, **kwargs):
    tensor = kwargs.get("tensor", False)
    if all(isinstance(matrix, MutableDenseMatrix) for matrix in matrices):
        return matrices
    size = 0
    zeros = 0
    dim = 1
    for matrix in matrices:
        size += matrix.shape[0] * matrix.shape[1]
        zeros += count_zeros(matrix)
        dim *= matrix.shape[0]
    zero_fraction = zeros / size
    if tensor == False:
        dim = matrices[0].shape[0]
    matrix_list = []
    if zero_fraction >= sparse_matrix_threshold and dim > sparse_threshold or dim > dense_limit:
        for mat in matrices:
            if sparse.issparse(mat):
                matrix_list.append(mat)
            else:
                matrix_list.append(sparse.csr_matrix(mat, dtype=np.complex128))
        return matrix_list
    else:
        for mat in matrices:
            if sparse.issparse(mat):
                matrix_list.append(np.asarray(mat.todense(), dtype=np.complex128))
            else:
                matrix_list.append(np.asarray(mat, dtype=np.complex128))
        return matrix_list
  

@log_function_call
def convert_to_sparse(matrix):
    if sparse.issparse(matrix) or isinstance(matrix, MutableDenseMatrix):
        return matrix
    zero_fraction = count_zeros(matrix) / matrix.size
    if zero_fraction >= sparse_matrix_threshold and matrix.shape[0] > sparse_threshold:
        return sparse.csr_matrix(matrix, dtype=np.complex128) 
    return matrix


@log_function_call
def convert_to_sparse_array(array):
    if sparse.issparse(array):
        return array
    zero_fraction = count_zeros(array) / array.size
    if zero_fraction >= sparse_matrix_threshold and array.shape[0]**2 > sparse_threshold:
        return sparse.csr_array(array, dtype=np.complex128)
    return array


@log_function_call
def convert_to_dense(matrix):
    if isinstance(matrix, np.ndarray) or isinstance(matrix, MutableDenseMatrix):
        return matrix
    zero_fraction = count_zeros(matrix) / matrix.size
    if zero_fraction <= sparse_matrix_threshold and sparse.issparse(matrix):
        return np.asarray(matrix.todense(), dtype=np.complex128)
    
@log_function_call
def convert_to_dense_array(array):
    if isinstance(array, np.ndarray):
        return array.ravel()
    zero_fraction = count_zeros(array) / array.size
    if zero_fraction <= sparse_matrix_threshold and sparse.issparse(array):
        return np.asarray(array.todense(), dtype=np.complex128)
    

@log_function_call
def sparse_mat(matrix):
    if sparse.issparse(matrix) or isinstance(matrix, MutableDenseMatrix):
        logging.debug(f"Leaving sparse matrix or sympy matrix as is")
        return matrix
    else:
        logging.debug(f"Converting all other types to sparce.csr_matrix")
        return sparse.csr_matrix(matrix, dtype=np.complex128)
    

@log_function_call
def sparse_array(array):
    if sparse.issparse(array):
        logging.debug(f"Leaving sparse array as is")
        return array
    else:
        logging.debug(f"Converting all other types to sparse.scr_array")
        return sparse.csr_array(array, dtype=np.complex128)


@log_function_call
def dense_mat(matrix):
    if sparse.issparse(matrix):
        logging.debug(f"Converting sparse matrix to a numpy matrix")
        return np.asarray(matrix.todense(), dtype=np.complex128)
    elif isinstance(matrix, MutableDenseMatrix):
        logging.debug(f"Leaving sympy matrix as is")
        return matrix
    elif isinstance(matrix, np.ndarray):
        logging.debug(f"Converting numpy matrix to numpy matrix")
        return np.asarray(matrix, dtype=np.complex128)
    elif isinstance(matrix, list):
        logging.debug(f"Converting list to numpy matrix")
        return np.array(matrix, dtype=np.complex128)
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(matrix)}")


@log_function_call
def dense_array(array):
    if isinstance(array, sparse.spmatrix): 
        logging.debug(f"Converting sparse array to a numpy array")
        return np.asarray(array.todense(), dtype=np.complex128)
    elif isinstance(array, np.ndarray):
        logging.debug(f"Converting numpy array to numpy array")
        return np.asarray(array, dtype=np.complex128).ravel()
    elif isinstance(array, list):
        logging.debug(f"Converting list to numpy array")
        return np.array(array, dtype=np.complex128).ravel()
    raise SparseMatrixError(f"Expected sparse matrix or ndarray, got {type(array)}")