import numpy as np
import sympy as sp

sp.init_printing(use_unicode=True)

class Gate:
    def __init__(self, name, a, b, c, d):
        self.name = name
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __str__(self):
        return f"{self.name}({self.a},{self.b})({self.c},{self.d})"

    def norm(self):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        norm = sp.sqrt(a**2 + b**2 + c**2 + d**2)
        self.a = a / norm
        self.b = b / norm
        self.c = c / norm
        self.d = d / norm
        self.a = sp.simplify(self.a)
        self.b = sp.simplify(self.b)
        self.c = sp.simplify(self.c)
        self.d = sp.simplify(self.d)
        

Hadamard = Gate("Hadamard", 1,1,1,-1)
Hadamard.norm()
print(Hadamard)
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





