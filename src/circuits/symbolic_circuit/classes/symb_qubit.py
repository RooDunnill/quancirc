import numpy as np
from ...base_classes.base_qubit import *
from ..utilities.circuit_errors import SymbQuantumStateError, SymbStatePreparationError
from ...circuit_config import *
import sympy as sp
from ..static_methods.symb_qubit_methods import *

__all__ = ["SymbQubit", "q0_symb", "q1_symb", "qp_symb", "qm_symb", "qpi_symb", "qmi_symb", "qgen_symb"]


class SymbQubit(BaseQubit):
    def __init__(self, **kwargs):
        object.__setattr__(self, 'class_type', 'symbqubit')
        object.__setattr__(self, "state_type", "Symbolic")
        super().__init__(**kwargs)
        self.rho = kwargs.get("rho", None)
        self.rho = sp.Matrix(self.rho) if self.rho is not None else None
        self.state = sp.Matrix(self.state) if self.state is not None else None
        self.weights = kwargs.get("weights", None)
        self.rho_init()
        self.dim = self.rho.shape[0]
        self.length = self.dim ** 2
        self.n = int(np.log2(self.dim))


    def __str__(self):
        return f"{self.name}:\n{self.rho}"
    
    def subs(self, substitution: dict) -> "SymbQubit":
        new_rho = self.rho.subs(substitution)
        kwargs = {"rho": new_rho}
        return SymbQubit(**kwargs)

    def __sub__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":
        """Subtraction of two SymbQubit rho matrices, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = self.rho - other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}             
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __add__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":
        """Addition of two SymbQubit rho matrices, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            print(f"1st Rho: {self.rho}")
            print(f"2nd Rho: {other.rho}")
            new_rho = self.rho + other.rho
            kwargs = {"rho": new_rho, "skip_validation": True}
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def __matmul__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":    
        """Matrix multiplication between two SymbQubit objects, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = self.rho * other.rho
            kwargs = {"rho": new_rho}
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"Objects cannot have types: {type(self)} and {type(other)}, expected type SymbQubit")
    
    def __mod__(self: "SymbQubit", other: "SymbQubit") -> "SymbQubit":
        """Tensor product among two SymbQubit objects, returns a SymbQubit object"""
        if isinstance(other, SymbQubit):
            new_rho = sp.kronecker_product(self.rho, other.rho)
            kwargs = {"rho": new_rho}
            return SymbQubit(**kwargs)
        raise SymbQuantumStateError(f"The classes do not match or the array is not defined. They are of types {type(self)} and {type(other)}")
    
    def rho_init(self):
        if self.rho is None:
                if self.weights is not None:
                    self.rho = self.build_mixed_rho()
                else:
                    self.rho = self.build_pure_rho()
    
    def build_mixed_rho(self):       #this is wrong
        """Builds a mixed rho matrix, primarily in initiation of Qubit object, returns type sp.MatrixBase"""
        if self.weights is not None:
            dim = self.state.shape[1]
            mixed_rho = sp.zeros(dim, dim)
            for i in range(len(self.weights)):
                state = sp.Matrix(self.state[i*dim:i*dim+dim])
                mixed_rho += self.weights[i] * (state * state.H)
            return mixed_rho
        raise SymbStatePreparationError(f"For a mixed rho to be made, you must provide weights in kwargs")
    
    @classmethod
    def create_mixed_state(self, states, weights):
        """This is used for when you want to combine premade states into a larger mixed state"""
        if not all(isinstance(obj, list) for obj in (states, weights)):
            raise SymbStatePreparationError(f"states and weights cannot be of type, {type(states)} and {type(weights)}, must be of type list and list")
        if not all(isinstance(state, SymbQubit) for state in states) or not all(isinstance(probs, (float, sp.Symbol)) for probs in weights):
            raise SymbStatePreparationError(f"States and weights must be made up of types SymbQubit and types float or sp.symbol")
        if len(states) != len(weights):
            raise SymbStatePreparationError(f"The amount of states must match the amount of weights given, not {len(states)} and {len(weights)}")
        new_rho = sp.zeros((states[0].dim, states[0].dim))
        for state, weight in zip(states, weights):
            new_rho += weight * state.rho
        kwargs = {"rho": new_rho}
        return SymbQubit(**kwargs)
    
    @classmethod
    def q0(cls, **kwargs):
        return q0_state(cls, **kwargs)

    @classmethod
    def q1(cls, **kwargs):
        return q1_state(cls, **kwargs)

    @classmethod
    def qp(cls):
        return qp_state(cls)

    @classmethod
    def qm(cls):
        return qm_state(cls)

    @classmethod
    def qpi(cls):
        return qpi_state(cls)

    @classmethod
    def qmi(cls):
        return qmi_state(cls)
    
    @classmethod
    def gen(cls, **kwargs):
        return gen_state(cls, **kwargs)
    

qgen_symb = SymbQubit.gen()    
q0_symb = SymbQubit.q0()
q1_symb = SymbQubit.q1()
qp_symb = SymbQubit.qp()
qm_symb = SymbQubit.qm()
qpi_symb = SymbQubit.qpi()
qmi_symb = SymbQubit.qmi()

