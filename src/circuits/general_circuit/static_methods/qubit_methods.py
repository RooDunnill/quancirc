import numpy as np

def q0_state(cls, **kwargs):
    """The |0> Qubit"""
    n = kwargs.get("n", 1)
    q0_vector = [1,0]
    q0_id = f"|0>"
    if n != 1:
        q0_vector = np.zeros(2**n, dtype=np.complex128)
        q0_vector[0] = 1
        zeros_str = "0" * n
        q0_id = f"|{zeros_str}>"
    return cls(id=q0_id, state=q0_vector)

def q1_state(cls, **kwargs):
    """The |1> Qubit"""
    n = kwargs.get("n", 1)
    q1_vector = [0,1]
    q1_id = f"|1>"
    if n != 1:
        q1_vector = np.zeros(2**n, dtype=np.complex128)
        q1_vector[-1] = 1
        ones_str = "1" * n
        q1_id = f"|{ones_str}>"
    return cls(id=q1_id, state=q1_vector)

def qp_state(cls):
    """The |+> Qubit"""
    n = 1/np.sqrt(2)
    qp_vector = [n,n]
    return cls(id="|+>", state=qp_vector)

def qm_state(cls):
    """The |-> Qubit"""
    n = 1/np.sqrt(2)
    qm_vector = [n,-n]
    return cls(id="|->", state=qm_vector)

def qpi_state(cls):
    """The |i> Qubit"""
    n =1/np.sqrt(2)
    qpi_vector = np.array([n+0j,0+n*1j],dtype=np.complex128)
    return cls(id="|i>", state=qpi_vector)

def qmi_state(cls):
    """The |-i> Qubit"""
    n =1/np.sqrt(2)
    qmi_vector = np.array([n+0j,0-n*1j],dtype=np.complex128)
    return cls(id="|-i>", state=qmi_vector)