import numpy as np
import sympy as sp
import random as rm
sp.init_printing(use_unicode=True)

class Gate_data:
    C_Not_info = "test"
    C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]
    X_Gate_info = "test"
    X_matrix = [0,1,1,0]
    Y_Gate_info = "test"
    Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]
    n = 1/sp.sqrt(2)
    Hadamard_info = "test"
    Hadamard_matrix = [n,n,n,-n]
    Z_Gate_info = "test"
    Z_matrix = [1,0,0,-1]
    Identity_info = """
Identity Matrix: This matrix leaves the product invariant after multiplication.
It is mainly used in this program to increase the dimension
of other matrices. This is used within the tensor products when
a Qubit has no gate action, but the others do.
"""
    Identity_matrix = [1,0,0,1]
    operation_error = "This is not a valid operation"
    




class Qubit:
    def __init__(self, name, vector):
        self.name = name
        self.vector = np.array(vector,dtype=np.complex128)
        self.dim = len(vector)
        

    def __str__(self):
        return f"{self.name}\n{self.vector}"
    
    def __matmul__(self, other):
        if isinstance(other, Qubit):
            new_name = f"{self.name} @ {other.name}"
            new_length = self.dim*other.dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):
                for j in range(other.dim):
                    new_vector[j+i*other.dim] += self.vector[i]*other.vector[j]
            sp.simplify(new_vector)
            return Qubit(new_name, np.array(new_vector))
        else:
            Print(Gate_data.operation_error)


    def norm(self):
        normalise = sp.sqrt(sum([i*np.conj(i) for i in self.vector]))
        self.vector = sp.simplify(self.vector/normalise)

    def qubit_info(self):

        pass


    def measure(self, qubit):
        if qubit <= self.dim:
            qubit_choice = [0,1]
            a = np.complex128(self.vector[2*qubit-2]*np.conj(self.vector[2*qubit-2]))
            b = np.complex128(self.vector[2*qubit-1]*np.conj(self.vector[2*qubit-1]))
            print(a)
            print(b)
            prob_mat = np.array([a,b],dtype=np.complex128)
            measurement = rm.choices(qubit_choice,weights = prob_mat)
            return measurement
        else:
            print(Gate_data.operation_error)


        pass

q0_matrix = [1,0]      
q1_matrix = [0,1]
qplus_matrix = [1,1]
qminus_matrix = [1,-1]
q0 = Qubit("q0",q0_matrix)
q1 = Qubit("q1",q1_matrix)
qplus = Qubit("q+",qplus_matrix)
qminus = Qubit("q-",qminus_matrix)
q0.norm()
q1.norm()
qplus.norm()
qminus.norm()

class Gate:
    def __init__(self, name, info, matrix):
        self.name = name
        self.matrix = np.array(matrix,dtype=np.complex128)
        self.info = info
        self.length = len(matrix)
        self.dim = int(sp.sqrt(self.length))
        

    def __str__(self):
        return f"{self.name}\n{self.matrix}"

    def __matmul__(self, other):
        if isinstance(other, Gate):
            new_info = "This is a tensor product of gates: "f"{self.name}"" and "f"{other.name}"
            new_name = f"{self.name} @ {other.name}"
            new_length = self.length*other.length
            new_dim = sp.sqrt(new_length)
            new_mat = np.zeros(new_length,dtype=np.complex128)
            for m in range(self.dim):
                for i in range(self.dim):
                    for j in range(other.dim):
                        for k in range(other.dim):
                            new_mat[k+j*new_dim+other.dim*i+other.dim*new_dim*m] += self.matrix[i+self.dim*m]*other.matrix[k+other.dim*j]
            sp.simplify(new_mat)
            return Gate(new_name, new_info, np.array(new_mat))
        else:
            print(Gate_data.operation_error)

    def __mul__(self, other):
        summ = np.zeros(1,dtype=np.complex128)
        if isinstance(other, Gate):
            if self.dim == other.dim:
                new_info = "This is a matrix multiplication of gates: "f"{self.name}"" and "f"{other.name}"
                new_name = f"{self.name} * {other.name}"
                new_mat = np.zeros(self.length,dtype=np.complex128)
                for i in range(self.dim):
                    for k in range(self.dim):
                        for j in range(self.dim):
                            summ[0] += (self.matrix[j+self.dim*i]*other.matrix[k+j*self.dim])
                        new_mat[k+self.dim*i] += summ[0]
                        summ = np.zeros(1,dtype=np.complex128)
                sp.simplify(new_mat)
                return Gate(new_name, new_info, np.array(new_mat))
            else:
                print(Gate_data.operation_error)
        elif isinstance(other, Qubit):
            new_name = f"[{self.name}] {other.name}"
            new_mat = np.zeros(self.dim,dtype=np.complex128)
            for i in range(self.dim):
                    for j in range(self.dim):
                        summ[0] += (self.matrix[j+self.dim*i]*other.vector[j])
                    new_mat[i] += summ[0]
                    summ = np.zeros(1,dtype=np.complex128)
            return Qubit(new_name, np.array(new_mat))
        else:
            print(Gate_data.operation_error)
    
    def __add__(self, other):
        if isinstance(other, Gate):
            new_info = "This is a direct sum of gates: "f"{self.name}"" and "f"{other.name}"
            new_name = f"{self.name} + {other.name}"
            new_dim = self.dim + other.dim
            new_length = new_dim**2
            new_mat = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):
                for j in range(self.dim):
                    new_mat[j+new_dim*i] += self.matrix[j+self.dim*i]
            for i in range(other.dim):
                for j in range(other.dim):
                    new_mat[self.dim+j+self.dim*new_dim+new_dim*i] += other.matrix[j+other.dim*i]
            return Gate(new_name, new_info, np.array(new_mat))
        else:
            print(Gate_data.operation_error)

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







C_Not = Gate("C_Not", Gate_data.C_Not_info, Gate_data.C_Not_matrix)
X_Gate = Gate("X", Gate_data.X_Gate_info, Gate_data.X_matrix)
Y_Gate = Gate("Y",Gate_data.Y_Gate_info, Gate_data.Y_matrix)
Z_Gate = Gate("Z",Gate_data.Z_Gate_info, Gate_data.Z_matrix)
Identity = Gate("I",Gate_data.Identity_info, Gate_data.Identity_matrix)
Hadamard = Gate("H",Gate_data.Hadamard_info, Gate_data.Hadamard_matrix)
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
test_qubit = qplus @ qminus @ q0
test_state = Y_Gate @ Hadamard + Y_Gate
test_measure = test_state * test_qubit
print(test_measure)
print(test_measure.measure(2))