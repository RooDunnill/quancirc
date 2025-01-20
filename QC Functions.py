import numpy as np
import sympy as sp

sp.init_printing(use_unicode=True)

class Gate:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = np.array(matrix)

    def __str__(self):
        return f"{self.name}({self.matrix})"

    def norm(self):
        normalise = sp.sqrt(sum([i**2 for i in self.matrix]))
        self.matrix = sp.simplify(self.matrix/normalise)

    def gate_mult(self, other):
        new_name = f"{self.name} {other.name}"
        length = len(self.matrix)
        dim = sp.sqrt(len(self.matrix))
        new_mat = []
        summ = 0
        for i in range(dim):
            for k in range(dim):
                for j in range(dim):
                    summ += (self.matrix[j+dim*i]*other.matrix[k+j*dim])
                print(summ)
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


class Qubit:
    def __init__(self, name, a, b):
        self.name = name
        self.a = a
        self.b = b

    def __str__(self):
        return f"{self.name}({self.a},{self.b})"
    def tensor(self, q):
        pass
        

    def norm(self):
        a = self.a
        b = self.b
        self.a = a / sp.sqrt(a**2 + b**2)
        self.b = b / sp.sqrt(a**2 + b**2)
        self.a = sp.simplify(self.a)
        self.b = sp.simplify(self.b)
        
        


    def measure(self):

        pass
        

q1 = Qubit("q1",1,2)
q2 = Qubit("q2",0,1)
q1.norm()
q2.norm()
print(q1,q2)





