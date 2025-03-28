import numpy as np
from scipy.sparse import eye_array


def identity_gate(cls, **kwargs):    
    """The identity matrix, used mostly to represent empty wires in the circuit
    Args:
        n: int: creates an Identity matrix for n qubits, default is 1 Qubit
    Returns:
        Gate, the identity gate with either custom qubits or for a single Qubit"""
    n = kwargs.get("n", 1)
    mat_type = kwargs.get("type", "dense")
    if isinstance(n, int):
        dim = int(2**n)
        if mat_type == "dense":
            new_mat = np.eye(dim, dtype=np.complex128)
        elif mat_type == "sparse":
            new_mat = eye_array(dim, dtype=np.complex128)
        else:
            raise GateError(f"mat_type cannot be {mat_type}, expected either 'sparse' or 'dense'")
        return cls(name="Identity Gate", matrix=new_mat)
    
def rotation_x_gate(cls, theta):
    """Rotates a qubit around the x axis"""
    rotation_x = [[np.cos(theta/2),np.complex128(0-1j)*np.sin(theta/2)],[np.complex128(0-1j)*np.sin(theta/2),np.cos(theta/2)]]
    return cls(name="Rotation X", matrix=rotation_x)

def rotation_y_gate(cls, theta):
    """Rotates a qubit around the x axis"""
    rotation_y = [[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]
    return cls(name="Rotation Y", matrix=rotation_y)

def rotation_z_gate(cls, theta):
    """Rotates a qubit around the x axis"""
    rotation_z = [[np.exp(np.complex128(0-1j)*(theta/2)),0],[0,np.exp(np.complex128(0+1j)*(theta/2))]]
    return cls(name="Rotation Z", matrix=rotation_z)

def pauli_x_gate(cls):
    """The X Gate, which can flip the Qubits in the X or computational basis"""
    X_matrix = [[0,1],[1,0]]
    return cls(name="X Gate", matrix=X_matrix)

def pauli_y_gate(cls):
    """The Y Gate, which can flip the Qubits in the Y basis"""
    Y_matrix = [[0,np.complex128(0-1j)],[np.complex128(0+1j),0]]
    return cls(name="Y Gate", matrix=Y_matrix)

def pauli_z_gate(cls):
    """The Z Gate, which can flip the Qubits in the Z basis or |+>, |-> basis"""
    Z_matrix = [[1,0],[0,-1]]
    return cls(name="Z Gate", matrix=Z_matrix)

def hadamard_gate(cls):
    """THe Hadamard Gate, commonly used to rotate between the computational or X basis and the |+>, |-> or Z basis"""
    n = 1/np.sqrt(2)
    H_matrix = [[n,n],[n,-n]]
    return cls(name="Hadamard", matrix=H_matrix)

def phase_gate(cls, theta, **kwargs):
    """The phase Gate used to add a local phase to a Qubit"""
    name = kwargs.get("name", f"Phase Gate with a phase {theta:.3f}")    #allows custom naming to creats S and T named gates
    P_matrix = [[1,0],[0,np.exp(1j*theta)]]
    return cls(name=name, matrix=P_matrix)

def unitary_gate(cls, a, b, c):
    """The Unitary Gate, which can approximate nearly any unitary 2 x 2 Gate"""
    U_matrix = [[np.cos(a/2),
                -np.exp(np.complex128(0-1j)*c)*np.sin(a/2)],
                [np.exp(np.complex128(0+1j)*b)*np.sin(a/2),                  #not hugely used, just thought it was cool to implement
                np.exp(np.complex128(0+1j)*(b+c))*np.cos(a/2)]]
    return cls(name=f"Unitary Gate with values (a:{a:.3f}, b:{b:.3f}, c:{c:.3f})", matrix=U_matrix)
    
def swap_gate(cls, **kwargs):
    """The Swap Gate, used to flip two Qubits in a circuit"""
    n_is_2 = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
    n: int = kwargs.get("n", 2)
    if n == 2:
        return cls(name=f"2 Qubit Swap gate", matrix=n_is_2)
    
