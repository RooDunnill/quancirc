import numpy as np
from utilities import QuantumStateError, StatePreparationError
from .class_methods.qubit_methods import *


class Qubit:                                           #creates the qubit class
    """The class to define and initialise Qubits and Quantum States"""

    def __init__(self, **kwargs) -> None:
        self.name: str = kwargs.get("name", None)
        self.state_type: str = kwargs.get("type", "pure")                   #the default qubit is a single pure qubit |0>
        self.display_mode = "vector"
        self.weights: list = kwargs.get("weights", None)
        self.name: str = kwargs.get("name","|Quantum State>")
        self.state: list = kwargs.get("state", None)
        self.rho: list = kwargs.get("rho", None)

        if self.state is not None:
            if not isinstance(self.state, (list, np.ndarray)):
                raise StatePreparationError(f"The inputted self.state cannot be of type {type(self.state)}, expected list or np.ndarray")
            self.state = np.array(self.state, dtype=np.complex128)

        if self.state is not None and self.rho is None:
            if self.state_type == "pure":
                self.rho = self.build_pure_rho()
            elif self.state_type == "mixed":
                self.rho = self.build_mixed_rho()

        if self.rho is not None:
            if not isinstance(self.rho, (list, np.ndarray)):
                raise StatePreparationError(f"The inputted self.rho cannot be of type {type(self.rho)}, expected list or np.ndarray")
            self.rho = np.array(self.rho, dtype=np.complex128)
        else:
            raise StatePreparationError(f"The initialised object must have atleast 1 of the following: a state vector or a density matrix")

        self.length = len(self.rho)
        self.dim = int(np.sqrt(self.length))
        self.n = int(np.log2(self.dim))


    def __str__(self):
        if self.display_mode == "vector":
            return f"Quantum State Vector: {self.build_pure_state()}"
        elif self.display_mode == "density":
            return f"Quantum State Density Matrix: {self.rho}"
        elif self.display_mode == "both":
            return f"Quantum State Vector:\n {self.build_pure_state()}\n and Density Matrix:\n {self.rho}"
        

    def combine_qubit_attr(self, other, op = "+"):
        """Allows the returned objects to still return name and info too"""
        kwargs = {}
        if op == "@":
            self_name_size = int(np.log2(self.dim))
            other_name_size = int(np.log2(other.dim)) 
            kwargs["name"] = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>"
        elif hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            kwargs["name"] = f"{self.name} {op} {other.name}"
        if hasattr(self, "state_type") and hasattr(other, "state_type"):
            if self.state_type == "mixed" or other.state_type == "mixed":
                kwargs["state_type"] = "mixed"
            elif self.state_type == "pure" or other.state_type == "pure":
                kwargs["state_type"] = "pure"
            else:
                raise QuantumStateError(f"The state types must be either 'pure' or 'mixed', not {self.state_type} and {other.state_type}")
        return kwargs
        
    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            new_rho = np.kron(self.rho, other.rho)
            kwargs = {"rho": new_rho}
            kwargs.update(self.combine_qubit_attr(other, op = "@"))
            return self.__class__(**kwargs)
        else:
            raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
        
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            raise QuantumStateError(f"Cannot matrix multiply two Quantum states together")
        else:
            new_rho = np.dot(self.rho)
            


    def __sub__(self, other):
        if isinstance(other, self.__class__):
            new_rho = self.rho - other.rho
            kwargs = {"rho": new_rho}
            kwargs.update(self.combine_qubit_attr(other, op = "-"))
            return self.__class__(**kwargs)
        else:
            raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_rho = self.rho + other.rho
            kwargs = {"rho": new_rho}
            kwargs.update(self.combine_qubit_attr(other, op = "+"))
            return self.__class__(**kwargs)
        else:
            raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __and__(self, other):
        if isinstance(other, self.__class__):
            new_rho = np.block([[self.rho, np.zeros_like(other.rho)], [np.zeros_like(self.rho), other.rho]])
            kwargs = {"rho": new_rho}
            kwargs.update(self.combine_qubit_attr(other, op = "&"))
            return self.__class__(**kwargs)
        else:
            raise QuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __eq__(self, other):
        return self.rho == other.rho
    
    def norm(self):
        trace_rho = np.trace(self.rho)
        if trace_rho != 0:
            self.rho = self.rho / trace_rho
        else:
            raise QuantumStateError(f"The trace of the density matrix cannot be 0 and so cannot normalise")

    def set_display_mode(self, mode):
        if mode not in ["vector", "density", "both"]:
            raise QuantumStateError(f"The display mode must be set in 'vector', 'density' or 'both'")
        self.display_mode = mode

    def build_pure_rho(self):
        return np.outer(np.conj(self.state), self.state)
    
    def build_mixed_rho(self):
        if self.weights is not None:
            mixed_rho = np.zeroes(len(self.state[0])**2, dtype=np.complex128)
            for i in range(len(self.weights)):
                mixed_rho += self.weights[i] * np.outer(np.conj(self.state[i]), self.state[i])
            return mixed_rho
        
    def build_pure_state(self):
        return np.diag(self.rho)

    def build_mixed_state(self):
        probs, states = np.linalg(self.rho)
        return probs, states
    
    @classmethod
    def q0(cls, **kwargs):
        return q0(cls, **kwargs)

    @classmethod
    def q1(cls, **kwargs):
        return q1(cls, **kwargs)

    @classmethod
    def qp(cls):
        return qp(cls)

    @classmethod
    def qm(cls):
        return qm(cls)

    @classmethod
    def qpi(cls):
        return qpi(cls)

    @classmethod
    def qmi(cls):
        return qmi(cls)
    
q0 = Qubit.q0()
q1 = Qubit.q1()
qp = Qubit.qp()
qm = Qubit.qm()
qpi = Qubit.qpi()
qmi = Qubit.qmi()

