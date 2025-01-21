import numpy as np
import sympy as sp
import random as rm
sp.init_printing(use_unicode=True)

class Gate_data:
    pass
    




class Qubit:
    def __init__(self, name, vector):
        self.name = name
        self.vector = np.array(vector,dtype=np.complex128)
        self.dim = len(vector)
        

    def __str__(self):
        return f"{self.name}\n{self.vector}"
    
    def __matmul__(self, other):
        if isinstance(other, Qubit):
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

    def Qubit_info(self):

        pass


    def measure(self):

        pass

q0_matrix = [1,0]      
q1_matrix = [0,1]

q0 = Qubit("q0",q0_matrix)
q1 = Qubit("q1",q1_matrix)
q0.norm()
q1.norm()


class Gate:
    def __init__(self, name, info, matrix):
        self.name = name
        self.matrix = np.array(matrix,dtype=np.complex128)
        self.info = info
        self.dim = int(sp.sqrt(len(matrix)))

    def __str__(self):
        return f"{self.name}\n{self.matrix}"

    def __matmul__(self, other):
        if isinstance(other, Gate):
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
        return Gate(new_name, self.info, np.array(new_mat))

    def __mul__(self, other):
        length = len(self.matrix)
        dim = int(sp.sqrt(len(self.matrix)))
        summ = np.zeros(1,dtype=np.complex128)
        if isinstance(other, Gate):
            new_name = f"{self.name} * {other.name}"
            new_mat = np.zeros(length,dtype=np.complex128)
            for i in range(dim):
                for k in range(dim):
                    for j in range(dim):
                        summ[0] += (self.matrix[j+dim*i]*other.matrix[k+j*dim])
                    new_mat[k+dim*i] += summ[0]
                    summ = np.zeros(1,dtype=np.complex128)
            sp.simplify(new_mat)
            return Gate(new_name, self.info, np.array(new_mat))
        elif isinstance(other, Qubit):
            new_name = f"[{self.name}] {other.name}"
            
            new_mat = np.zeros(dim,dtype=np.complex128)
            for i in range(dim):
                    for j in range(dim):
                        summ[0] += (self.matrix[j+dim*i]*other.vector[j])
                    new_mat[i] += summ[0]
                    summ = np.zeros(1,dtype=np.complex128)
            return Qubit(new_name, np.array(new_mat))
    def gate_info(self):
        print(
    """Gates are used to apply an operation to a Qubit.
    They are normally situated on a grid of n Qubits.
    Using tensor products, we can combine all the gates 
    at one time instance together to create one unitary matrix.
    Then we can matrix multiply successive gates together to creat one
    universal matrix that we can apply to the Qubit before measuring""")

class print_array:
    def __init__(self, array):
        self.array = array
        prec = 3
        if isinstance(array, Qubit):
            np.set_printoptions(precision=prec,linewidth=20,suppress=True,floatmode="fixed")
            print(array)
        elif isinstance(array, Gate):
            
            np.set_printoptions(precision=prec,linewidth=(3+2*(3+prec))*array.dim,suppress=True,floatmode="fixed")
            print(array)
        else:
            print("Not applicable")






C_Not_info = "test"
C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]
C_Not = Gate("C_Not", C_Not_info, C_Not_matrix)
X_Gate_info = "test"
X_matrix = [0,1,1,0]
X_Gate = Gate("X", X_Gate_info, X_matrix)
Y_Gate_info = "test"
Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]
Y_Gate = Gate("Y",Y_Gate_info, Y_matrix)
Z_Gate_info = "test"
Z_matrix = [1,0,0,-1]
Z_Gate = Gate("Z",Z_Gate_info, Z_matrix)
Identity_info = """
Identity Matrix: This matrix leaves the product invariant after multiplication.
It is mainly used in this program to increase the dimension
of other matrices. This is used within the tensor products when
a Qubit has no gate action, but the others do.
"""
Identity_matrix = [1,0,0,1]
Identity = Gate("I",Identity_info, Identity_matrix)
n = 1/sp.sqrt(2)
Hadamard_info = "test"
Hadamard_matrix = [n,n,n,-n]
Hadamard = Gate("H",Hadamard_info, Hadamard_matrix)
#print(Hadamard.gate_mult(X_Gate))
#print(Hadamard.gate_mult(Y_Gate))
#print(C_Not.tensor_prod(Hadamard))
#print(Hadamard @ X_Gate @ X_Gate)
#print(Hadamard * X_Gate * X_Gate)
#print(Hadamard*q1)
#print(Hadamard*Hadamard*q1)
#print(Identity.gate_info)

def Test_Alg(Qubit):
    gate1 = X_Gate @ C_Not
    gate2 = Hadamard @ Hadamard @ X_Gate
    gate3 = C_Not @ X_Gate
    alg = gate3 * gate2 * gate1
    result = alg * Qubit
Test_Alg(q1 @ q1 @ q0)
print_array(Identity @ Hadamard @ Identity)
print_array(Hadamard)
print_array(Hadamard @ Hadamard)
print_array(q1 @ q1)