import numpy as np
from scipy.linalg import expm
from scipy.sparse import eye_array
from ..utilities.circuit_errors import QutritGateError

def identity_gate(cls, **kwargs):    
    """The identity matrix, used mostly to represent empty wires in the circuit
    Args:
        n: int: creates an Identity matrix for n qubits, default is 1 Qubit
    Returns:
        Gate, the identity gate with either custom qubits or for a single Qubit"""
    n = kwargs.get("n", 1)
    mat_type = kwargs.get("type", "dense")
    if isinstance(n, int):
        dim = int(3**n)
        if mat_type == "dense":
            new_mat = np.eye(dim, dtype=np.complex128)
        elif mat_type == "sparse":
            new_mat = eye_array(dim, dtype=np.complex128)
        else:
            raise QutritGateError(f"mat_type cannot be {mat_type}, expected either 'sparse' or 'dense'")
        return cls(name="I" * n, matrix=new_mat)
    
def gell_mann_1(cls, theta, **kwargs):
    name = kwargs.get("name", "gm1({theta:.3f})")
    gm1_matrix = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm1_matrix * theta))

def gell_mann_2(cls, theta, **kwargs):
    name = kwargs.get("name", "gm2({theta:.3f})")
    gm2_matrix = np.array([[0,np.complex128(-1j),0],[np.complex128(1j),0,0],[0,0,0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm2_matrix * theta))

def gell_mann_3(cls, theta, **kwargs):
    name = kwargs.get("name", "gm3({theta:.3f})")
    gm3_matrix = np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm3_matrix * theta))

def gell_mann_4(cls, theta, **kwargs):
    name = kwargs.get("name", "gm4({theta:.3f})")
    gm4_matrix = np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm4_matrix * theta))

def gell_mann_5(cls, theta, **kwargs):
    name = kwargs.get("name", "gm5({theta:.3f})")
    gm5_matrix = np.array([[0,0,np.complex128(-1j)],[0,0,0],[np.complex128(1j),0,0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm5_matrix * theta))

def gell_mann_6(cls, theta, **kwargs):
    name = kwargs.get("name", "gm6({theta:.3f})")
    gm6_matrix = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm6_matrix * theta))

def gell_mann_7(cls, theta, **kwargs):
    name = kwargs.get("name", "gm7({theta:.3f})")
    gm7_matrix = np.array([[0,0,0],[0,0,np.complex128(-1j)],[0,np.complex128(1j),0]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm7_matrix * theta))

def gell_mann_8(cls, theta, **kwargs):
    name = kwargs.get("name", "gm8({theta:.3f})")
    n = 1/np.sqrt(3)
    gm8_matrix = np.array([[n,0,0],[0,n,0],[0,0,-2*n]], dtype=np.complex128)
    return cls(name=name, matrix=expm(1j * gm8_matrix * theta))

def phase_gate(cls, theta, phi, gamma, **kwargs):
    name = kwargs.get("name", f"P({theta:.3f}{phi:.3f}{gamma:.3f})")
    phase_matrix = [[np.exp(np.complex128(1j * theta)),0,0],[0,np.exp(np.complex128(1j * phi)),0],[0,0,np.exp(np.complex128(1j * gamma))]]
    return cls(name=name, matrix=phase_matrix)

def hadamard_uniform(cls):
    n = 1/np.sqrt(3)
    hadamard_matrix = np.array([[n,n,n],[n,n,n],[n,n,n]], dtype=np.complex128)
    return cls(name="Huni", matrix=hadamard_matrix)

def hadamard_dft(cls):
    n = 1/np.sqrt(3)
    omega = np.exp(np.complex128((2 * np.pi * 1j) / 3))
    hadamard_matrix = np.array([[1,1,1],[1, omega, omega**2],[1,omega**2, omega]], dtype=np.complex128)
    hadamard_matrix *= n
    return cls(name="Hdft", matrix=hadamard_matrix)

def hadamard_phase(cls, theta, **kwargs):
    name = kwargs.get("name", f"H({theta:.3f})")
    n = 1/np.sqrt(3)
    rotation = np.exp(np.complex128(1j * theta))
    hadamard_matrix = np.array([[1,1,1],[1,rotation,rotation],[1,rotation,rotation**2]], dtype=np.complex128)
    hadamard_matrix *= n
    return cls(name=name, matrix=hadamard_matrix)