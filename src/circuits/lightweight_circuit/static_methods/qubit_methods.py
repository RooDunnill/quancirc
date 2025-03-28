import numpy as np

def q0_state(cls, **kwargs):
    """The |0> Qubit"""
    n = kwargs.get("n", 1)
    q0_vector = [1,0]
    q0_name = f"|0>"
    if n != 1:
        q0_vector = np.zeros(2**n, dtype=np.complex128)
        q0_vector[0] = 1
        zeros_str = "0" * n
        q0_name = f"|{zeros_str}>"
    return cls(name=q0_name, state=q0_vector)

def q1_state(cls, **kwargs):
    """The |1> Qubit"""
    n = kwargs.get("n", 1)
    q1_vector = [0,1]
    q1_name = f"|1>"
    if n != 1:
        q1_vector = np.zeros(2**n, dtype=np.complex128)
        q1_vector[-1] = 1
        ones_str = "1" * n
        q1_name = f"|{ones_str}>"
    return cls(name=q1_name, state=q1_vector)

def qp_state(cls):
    """The |+> Qubit"""
    n = 1/np.sqrt(2)
    qp_vector = [n,n]
    return cls(name="|+>", state=qp_vector)

def qm_state(cls):
    """The |-> Qubit"""
    n = 1/np.sqrt(2)
    qm_vector = [n,-n]
    return cls(name="|->", state=qm_vector)

def qpi_state(cls):
    """The |i> Qubit"""
    n =1/np.sqrt(2)
    qpi_vector = np.array([n+0j,0+n*1j],dtype=np.complex128)
    return cls(name="|i>", state=qpi_vector)

def qmi_state(cls):
    """The |-i> Qubit"""
    n =1/np.sqrt(2)
    qmi_vector = np.array([n+0j,0-n*1j],dtype=np.complex128)
    return cls(name="|-i>", state=qmi_vector)