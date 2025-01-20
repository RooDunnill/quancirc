import numpy as np
import sympy as sp
sp.init_printing(use_unicode=True)

class Gate:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = np.array(matrix)

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
        new_mat = np.zeros(new_length)
        for m in range(self_dim):
            for i in range(self_dim):
                for j in range(other_dim):
                    for k in range(other_dim):
                        new_mat[k+j*new_dim+other_dim*i+other_dim*new_dim*m] += self.matrix[i+self_dim*m]*other.matrix[k+other_dim*j]
        return Gate(new_name, np.array(new_mat))

            



    def gate_mult(self, other):
        new_name = f"{self.name} x {other.name}"
        length = len(self.matrix)
        dim = sp.sqrt(len(self.matrix))
        new_mat = []
        summ = 0
        for i in range(dim):
            for k in range(dim):
                for j in range(dim):
                    summ += (self.matrix[j+dim*i]*other.matrix[k+j*dim])
                new_mat.append(summ)
                summ = 0
        print(new_mat)
        return Gate(new_name, np.array(new_mat))




C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]
C_Not = Gate("C_Not", C_Not_matrix)
X_matrix = [0,1,1,0]
X_matrix = Gate("X_Matrix", X_matrix)
Hadamard_matrix = [1,1,1,-1]
Hadamard = Gate("Hadamard", Hadamard_matrix)
print(Hadamard)
H2 = Hadamard.tensor_prod(Hadamard)
print(H2)
cn = C_Not.tensor_prod(Hadamard)
print(cn)



class Qubit:
    def __init__(self, name, a, b):
        self.name = name
        self.a = a
        self.b = b

    def __str__(self):
        return f"{self.name}({self.a},{self.b})"
    def tensor(self, q):
        pass
        
    #def norm(self):
     #   normalise = sp.sqrt(sum([i**2 for i in self.matrix]))
      #  self.matrix = sp.simplify(self.matrix/normalise)

        
        


    def measure(self):

        pass
        

q1 = Qubit("q1",1,2)
q2 = Qubit("q2",0,1)
print(q1,q2)





