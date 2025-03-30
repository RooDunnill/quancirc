import numpy as np
import sympy as sympy
from sympy import pprint
from ..circuits import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84
from ..crypto_protocols import otp
from ..crypto_protocols import rsa_weak_key_gen
from ..examples import *

print(Hadamard % Hadamard)
print(Hadamard_symb % Hadamard_symb)
print(q0_lw % q0_lw)
print(q0_symb % q0_symb)
print(q0 % q0)
print(X_Gate @ q0)
print(P_Gate_symb @ qg)
print(T_Gate @ qpi_lw)
print(Hadamard_symb % X_Gate_symb % T_Gate_symb)
print(CNot @ (q0 % q1))
print(Measure(state=q0).measure_state())
print(Measure(state=q1).measure_state())
print(Measure(state=qp).measure_state())
print(Measure(state=qm).measure_state())
print(Measure(state=qpi).measure_state())
print(Measure(state=qmi).measure_state())
print(LwMeasure(state=q0_lw).measure_state())
print(LwMeasure(state=q1_lw).measure_state())
print(LwMeasure(state=qp_lw).measure_state())
print(LwMeasure(state=qm_lw).measure_state())
print(LwMeasure(state=qpi_lw).measure_state())
print(LwMeasure(state=qmi_lw).measure_state())
