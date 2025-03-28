import numpy as np
from ..utilities.circuit_errors import SymbGateError
import sympy as sp


def identity_gate(cls, **kwargs):    
    """The identity matrix, used mostly to represent empty wires in the circuit
    Args:
        n: int: creates an Identity matrix for n qubits, default is 1 Qubit
    Returns:
        Gate, the identity gate with either custom qubits or for a single Qubit"""
    n = kwargs.get("n", 1)
    if isinstance(n, int):
        dim = int(2**n)
        new_mat = sp.eye(dim)
        return cls(name="Identity Gate", matrix=new_mat)
    raise SymbGateError(f"n cannot be {n}, expected type int")
        
    
def rotation_x_gate(cls, **kwargs):
    """Rotates a qubit around the x axis"""
    theta = kwargs.get("theta", sp.symbols("theta"))
    rotation_x = sp.Matrix([[sp.cos(theta/2),-sp.I*sp.sin(theta/2)],[-sp.I*sp.sin(theta/2),sp.cos(theta/2)]])
    return cls(name="Rotation X", matrix=rotation_x)

def rotation_y_gate(cls, **kwargs):
    """Rotates a qubit around the x axis"""
    theta = kwargs.get("theta", sp.symbols("theta"))
    rotation_y = sp.Matrix([[sp.cos(theta/2),-sp.sin(theta/2)],[sp.sin(theta/2),sp.cos(theta/2)]])
    return cls(name="Rotation Y", matrix=rotation_y)

def rotation_z_gate(cls, **kwargs):
    """Rotates a qubit around the x axis"""
    theta = kwargs.get("theta", sp.symbols("theta"))
    rotation_z = sp.Matrix([[sp.exp(-sp.I*(theta/2)),0],[0,sp.exp(sp.I*(theta/2))]])
    return cls(name="Rotation Z", matrix=rotation_z)

def pauli_x_gate(cls):
    """The X Gate, which can flip the Qubits in the X or computational basis"""
    X_matrix = sp.Matrix([[0,1],[1,0]])
    return cls(name="X Gate", matrix=X_matrix)

def pauli_y_gate(cls):
    """The Y Gate, which can flip the Qubits in the Y basis"""
    Y_matrix = sp.Matrix([[0,-sp.I],[sp.I,0]])
    return cls(name="Y Gate", matrix=Y_matrix)

def pauli_z_gate(cls):
    """The Z Gate, which can flip the Qubits in the Z basis or |+>, |-> basis"""
    Z_matrix = sp.Matrix([[1,0],[0,-1]])
    return cls(name="Z Gate", matrix=Z_matrix)

def hadamard_gate(cls):
    """THe Hadamard Gate, commonly used to rotate between the computational or X basis and the |+>, |-> or Z basis"""
    n = 1/sp.sqrt(2)
    H_matrix = sp.Matrix([[n,n],[n,-n]])
    return cls(name="Hadamard", matrix=H_matrix)

def phase_gate(cls, **kwargs):
    """The phase Gate used to add a local phase to a Qubit"""
    theta = kwargs.get("theta", sp.symbols("theta"))
    name = kwargs.get("name", f"Phase Gate with a phase {str(theta)}")    #allows custom naming to creats S and T named gates
    P_matrix = sp.Matrix([[1,0],[0,sp.exp(1j*theta)]])
    return cls(name=name, matrix=P_matrix)

def unitary_gate(cls, **kwargs):
    """The Unitary Gate, which can approximate nearly any unitary 2 x 2 Gate"""
    a = kwargs.get("a", sp.symbols("a"))
    b = kwargs.get("b", sp.symbols("b"))
    c = kwargs.get("c", sp.symbols("c"))
    U_matrix = sp.Matrix([[sp.cos(a/2),
                -sp.exp(-sp.I*c)*sp.sin(a/2)],
                [sp.exp(sp.I*b)*sp.sin(a/2),                  #not hugely used, just thought it was cool to implement
                sp.exp(sp.I*(b+c))*sp.cos(a/2)]])
    return cls(name=f"Unitary Gate with values (a:{str(a)}, b:{str(b)}, c:{str(c)})", matrix=U_matrix)
    
def swap_gate(cls, **kwargs):
    """The Swap Gate, used to flip two Qubits in a circuit"""
    n_is_2 = sp.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    return cls(name=f"2 Qubit Swap gate", matrix=n_is_2)
    
