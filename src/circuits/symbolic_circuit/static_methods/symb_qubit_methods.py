import sympy as sp


def gen_state(cls, **kwargs):
    """The basic general state"""
    alpha = kwargs.get("alpha", sp.symbols("alpha"))
    beta = kwargs.get("beta", sp.symbols("beta"))
    q_gen_vector = sp.Matrix([alpha, beta])
    q_gen_name = f"General Symbolic State"
    return cls(id=q_gen_name, state=q_gen_vector)

def q0_state(cls, **kwargs):
    """The |0> Qubit"""
    q0_vector = sp.Matrix([1,0])
    q0_name = f"|0>"
    return cls(id=q0_name, state=q0_vector)

def q1_state(cls, **kwargs):
    """The |1> Qubit"""
    q1_vector = sp.Matrix([0,1])
    q1_name = f"|1>"
    return cls(id=q1_name, state=q1_vector)

def qp_state(cls):
    """The |+> Qubit"""
    n = 1/sp.sqrt(2)
    qp_vector = sp.Matrix([n,n])
    return cls(id="|+>", state=qp_vector)

def qm_state(cls):
    """The |-> Qubit"""
    n = 1/sp.sqrt(2)
    qm_vector = sp.Matrix([n,-n])
    return cls(id="|->", state=qm_vector)

def qpi_state(cls):
    """The |i> Qubit"""
    n =1/sp.sqrt(2)
    qpi_vector = sp.Matrix([n+0j,0+n*1j])
    return cls(id="|i>", state=qpi_vector)

def qmi_state(cls):
    """The |-i> Qubit"""
    n =1/sp.sqrt(2)
    qmi_vector = sp.Matrix([n+0j,0-n*1j])
    return cls(id="|-i>", state=qmi_vector)