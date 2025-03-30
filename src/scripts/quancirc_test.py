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
normal_qubit_def = Qubit(state=[1,0])
print(normal_qubit_def)
normal_lw_qubit_def = LwQubit(state=[1,0])
print(normal_lw_qubit_def)
print(Qubit(rho=[[0.5,0],[0,0.5]]))
print(SymbQubit(state=[1,0]))
print(CNot @ (normal_lw_qubit_def % q0_lw))
print(CNot_flip @ (normal_qubit_def % qm))
normal_mixed_qubit_def = Qubit(state=[[1,0],[0,1],[np.sqrt(0.5),np.sqrt(0.5)]], weights=[0.2,0.5,0.3])
print(normal_mixed_qubit_def)
print(Hadamard @ normal_mixed_qubit_def)
large_qubit_def = q0 % q0 % qm % qm % qp % q0
large_qubit_state = (Hadamard % Hadamard % Hadamard % Hadamard % Hadamard % Hadamard) @ large_qubit_def
print(large_qubit_state[0])
print(large_qubit_state[1])
print(large_qubit_state[2])
print(large_qubit_state[3])
print(large_qubit_state[4])
print(large_qubit_state[5])
print(large_qubit_def.partial_trace(0,1))
print(large_qubit_def.partial_trace(0,2))
print(large_qubit_def.partial_trace(0,3))
print(large_qubit_def.partial_trace(0,4))
print(large_qubit_def.partial_trace(0,5))
print(large_qubit_def.partial_trace(1,1))
print(large_qubit_def.partial_trace(2,1))
print(large_qubit_def.partial_trace(3,1))
print(large_qubit_def.partial_trace(4,1))
print(large_qubit_def.partial_trace(0,2))
print(large_qubit_def.partial_trace(1,2))
print(large_qubit_def.partial_trace(2,2))
print(large_qubit_def.partial_trace(3,2))
print(large_qubit_def.partial_trace(0,3))
print(large_qubit_def.partial_trace(1,3))
print(large_qubit_def.partial_trace(2,3))
print(large_qubit_def.partial_trace(0,4))
print(large_qubit_def.partial_trace(1,4))
print(large_qubit_def.partial_trace(0,5))
large_qubit_def[0] = Hadamard @ large_qubit_def[0]
large_qubit_def[1] = Hadamard @ large_qubit_def[1]
large_qubit_def[2] = Hadamard @ large_qubit_def[2]
large_qubit_def[3] = Hadamard @ large_qubit_def[3]
large_qubit_def[4] = Hadamard @ large_qubit_def[4]
print(large_qubit_def.isolate_qubit(0))
print(large_qubit_def.isolate_qubit(1))
print(large_qubit_def.isolate_qubit(2))
print(large_qubit_def.isolate_qubit(3))
print(large_qubit_def.isolate_qubit(4))
print(q0 + q0)
print(q0 - q0)
large_qubit_def += large_qubit_def
print(large_qubit_def)
large_qubit_def -= large_qubit_def
print(large_qubit_def)
large_qubit_def @= large_qubit_def
print(large_qubit_def)
normal_mixed_qubit_def %= normal_mixed_qubit_def
normal_qubit_def %= normal_qubit_def
print(normal_mixed_qubit_def)
large_qubit_def.norm()
norm_test_1 = Qubit(state=[1,0])
norm_test_2 = Qubit(state=[0,1])
norm_test_1.norm()
norm_test_2.norm()
print(large_qubit_state)
large_qubit_state.norm()

