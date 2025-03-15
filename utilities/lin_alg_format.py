import numpy as np
from .qc_errors import QC_error

def trace(matrix) -> float:                 #used an np.trace doesnt calculate 1D matrices
    """Computes the trace of a 1D matrix, mostly used as a checker"""
    tr = 0
    if isinstance(matrix, np.ndarray):
        array = matrix
    elif hasattr(matrix, matrix.array_name):
        array = getattr(matrix, matrix.array_name)
    else:
        QC_error(f"The object {matrix} does not have an array assigned to it to be able to trace")
    dim = int(np.sqrt(len(array)))
    for i in range(dim):
        tr += array[i + i * dim]
    return tr
    
def reshape_matrix(matrix: np.ndarray) -> np.ndarray: 
    """Can reshape a 1D matrix into a square 2D matrix"""
    length = len(matrix)
    dim = int(np.sqrt(length))
    if dim**2 != length:
        raise QC_error(f"The matrix cannot be reshaped into a perfect square")
    reshaped_matrix = []
    for i in range(dim):
        row = matrix[i * dim : (i + 1) * dim]
        reshaped_matrix.append(row)
    return np.array(reshaped_matrix)

def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Can flatten a square 2D matrix into a 1D matrix"""
    dim = len(matrix)
    length = dim**2
    flattened_matrix = np.zeros(length, dtype=np.complex128)
    for j in range(dim):
        for i in range(dim):
            flattened_matrix[j * dim + i] += matrix[j][i]
    return flattened_matrix

def diagonal(matrix: np.ndarray) -> np.ndarray:                 
    """Creates a vector based on the idagonal elements of a 1D matrix"""
    dim = int(np.sqrt(len(matrix)))
    new_mat = np.zeros(dim, dtype=np.complex128)
    for i in range(dim):
        new_mat[i] = matrix[i+i*dim]
    return new_mat

def transpose(matrix: np.ndarray) -> np.ndarray: 
    if isinstance(matrix, np.ndarray):
        return flatten_matrix(reshape_matrix(matrix).T)
    raise QC_error(f"This cannot be of type {type(matrix)}, expected numpy array")

def adjoint(matrix: np.ndarray) -> np.ndarray:
    if isinstance(matrix, np.ndarray):
        return flatten_matrix(reshape_matrix(matrix).conj().T)
    raise QC_error(f"This cannot be of type {type(matrix)}, expected numpy array")