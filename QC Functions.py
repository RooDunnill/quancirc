import numpy as np
import sympy as sp
sp.init_printing(use_unicode=True)


class Qubit:
    def __init__(self, name, vector):
        self.name = name
        self.vector = np.array(vector,dtype=np.complex128)

    def __str__(self):
        return f"{self.name}({self.vector})"
    
    def tensor_prod(self, other):
        new_name = f"{self.name} X {other.name}"
        self_length = len(self.vector)
        other_length = len(other.vector)
        new_length = self_length*other_length
        new_vector = np.zeros(new_length,dtype=np.complex128)
        for i in range(self_length):
            for j in range(other_length):
                new_vector[j+i*other_length] += self.vector[i]*other.vector[j]
        sp.simplify(new_vector)
        return Qubit(new_name, np.array(new_vector))
    
    def norm(self):
        normalise = sp.sqrt(sum([i*np.conj(i) for i in self.vector]))
        self.vector = sp.simplify(self.vector/normalise)


    def measure(self):
        pass

q0_matrix = [1,0]      
q1_matrix = [0,1]

q0 = Qubit("q1",q0_matrix)
q1 = Qubit("q2",q1_matrix)
q0.norm()
q1.norm()
q3 = q1.tensor_prod(q0)
#print(q1,q0)
#print(q3)

class Gate:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = np.array(matrix,dtype=np.complex128)

    def __str__(self):
        return f"{self.name}({self.matrix})"

    def tensor_prod(self, other):
        new_name = f"{self.name} X {other.name}"
        self_length = len(self.matrix)
        self_dim = sp.sqrt(self_length)
        other_length = len(other.matrix)
        other_dim = sp.sqrt(other_length)
        new_length = self_length*other_length
        new_dim = sp.sqrt(new_length)
        new_mat = np.zeros(new_length,dtype=np.complex128)
        for m in range(self_dim):
            for i in range(self_dim):
                for j in range(other_dim):
                    for k in range(other_dim):
                        new_mat[k+j*new_dim+other_dim*i+other_dim*new_dim*m] += self.matrix[i+self_dim*m]*other.matrix[k+other_dim*j]
        sp.simplify(new_mat)
        return Gate(new_name, np.array(new_mat))

    def gate_mult(self, other):
        new_name = f"{self.name} x {other.name}"
        length = len(self.matrix)
        dim = sp.sqrt(len(self.matrix))
        new_mat = np.zeros(length,dtype=np.complex128)
        summ = np.zeros(1,dtype=np.complex128)
        for i in range(dim):
            for k in range(dim):
                for j in range(dim):
                    summ[0] += (self.matrix[j+dim*i]*other.matrix[k+j*dim])
                new_mat[k+dim*i] += summ[0]
                summ = np.zeros(1,dtype=np.complex128)
        sp.simplify(new_mat)
        return Gate(new_name, np.array(new_mat))




C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]
C_Not = Gate("C_Not", C_Not_matrix)
X_matrix = [0,1,1,0]
X_Gate = Gate("X_Gate", X_matrix)
Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]
Y_Gate = Gate("Y_Gate", Y_matrix)
Hadamard_matrix = [1,1,1,-1]
Hadamard = Gate("Hadamard", Hadamard_matrix)
#print(Hadamard.gate_mult(X_Gate))
#print(Hadamard.gate_mult(Y_Gate))
#print(C_Not.tensor_prod(Hadamard))

class Circuit:
    def __init__(self, name, matrix_array):
        self.name = name
        self.matrix_array = np.array(matrix_array,dtype=np.complex128)

    def __str__(self):
        return f"{self.name}({self.matrix_array})"
    
    def measure()

    def gate_mult(self):
        new_name = f"{self.name}"
        num_mats = len(self.matrix_array)
        matrix_length = int(sum(len(p) for p in self.matrix_array)/num_mats)
        dim = int(sp.sqrt(matrix_length))
        print(dim)
        new_mat = np.zeros(int(matrix_length),dtype=np.complex128)
        final_mat = np.zeros(int(matrix_length),dtype=np.complex128)
        summ = np.zeros(1,dtype=np.complex128)
        for m in range(num_mats-1):
            for i in range(dim):
                for k in range(dim):
                    for j in range(dim):
                        summ[0] += (self.matrix_array[m,j+dim*i]*self.matrix_array[m+1,k+j*dim])
                    new_mat[k+dim*i] += summ[0]
                    summ = np.zeros(1,dtype=np.complex128)
            for t in range(matrix_length):
                self.matrix_array[m+1,t]=new_mat[t]
            new_mat = np.zeros(int(matrix_length),dtype=np.complex128)
        print(self.matrix_array)
        for h in range(matrix_length):
            final_mat[h] = self.matrix_array[m+1,h]
        
        return Circuit(new_name, np.array(final_mat))
oneqcircuit_mat_array = [X_Gate.matrix,Hadamard.matrix,X_Gate.matrix]
oneqcircuit = Circuit("1 Qubit Circuit", oneqcircuit_mat_array)
print(oneqcircuit.gate_mult())






