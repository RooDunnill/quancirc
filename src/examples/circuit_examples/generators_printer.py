import numpy as np
from ...circuits.general_circuit.classes.gate import *
from ...circuits.circuit_utilities.sparse_funcs import *
pauli_gates = [Identity.matrix, X_Gate.matrix, Y_Gate.matrix, Z_Gate.matrix]

def remove_identity_matrices(matrix_list):
    identity_matrix = np.eye(matrix_list[0].shape[0]) 
    return [matrix for matrix in matrix_list if not np.array_equal(matrix, identity_matrix)]

def su_N_generators(N):
    generators = [Identity.matrix, X_Gate.matrix, Y_Gate.matrix, Z_Gate.matrix]
    for i in range(int(np.log2(N))-1):
        matrices = generators
        generators = []
        for j in matrices:
            for k in range(4):
                generators.append(np.kron(j,pauli_gates[k]))

    generators = remove_identity_matrices(generators)
    print(len(generators))
    return generators
    

