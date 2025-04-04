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
from ..circuits.circuit_utilities.sparse_funcs import *
from ..scripts.quancirc_checks import *
print(Hadamard % Hadamard)
print(Hadamard_symb % Hadamard_symb)
print(q0_lw % q0_lw)
print(q0_symb % q0_symb)
print(q0 % q0)
print(X_Gate @ q0)
print(P_Gate_symb @ qgen_symb)
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
large_qubit_state.debug()
large_qubit_def.debug()
large_qubit_state.norm()
complex_qub = Qubit.create_mixed_state([q0,qmi,qp,qm],[0.1,0.1,0.1,0.7])
complex_2_qub = Qubit.create_mixed_state([qpi,qpi,qmi,q0],[0.05,0.5,0.15,0.3])
print(complex_qub % complex_2_qub)
print(Hadamard @ complex_qub)
print(Swap @ (complex_2_qub % complex_qub))
complex_3_qub = Qubit.create_mixed_state([q0%q0,q0%q1,qm%qpi,qm%qm,qp%q1],[0.1,0.1,0.1,0.4,0.3])
complex_qub.debug()
complex_2_qub.debug()
complex_3_qub.debug()
print(complex_3_qub @ (complex_2_qub % complex_qub))
print(T_Gate @ qp)
t_gate_qub = Qubit(state=[0,0,0,0,0,0,0,1])
t_gate_qub = (Hadamard % Hadamard % Hadamard ) @ t_gate_qub

print((T_Gate % T_Gate % T_Gate) @ t_gate_qub)

t_gate_test = Gate(matrix=[[1,0],[0,np.exp(1j*np.pi/4)]])
t_gate_test.matrix = sparse_mat(t_gate_test.matrix)
print(t_gate_test @ q0)
q0.debug()

t_gate_circuit = Circuit(q=3)
t_gate_circuit.measure_states(qubit=0)
t_gate_circuit.apply_gate(Hadamard, qubit=1)
t_gate_circuit.apply_gate(Identity % T_Gate % Identity)

t_gate_circuit = Circuit(q=3)
t_gate_circuit.measure_states(qubit=0)
t_gate_circuit.apply_gate(Hadamard, qubit=1)
t_gate_circuit.apply_gate(T_Gate, qubit=1)

test_circuit = Circuit(q=3)
test_circuit.apply_gate(Hadamard, qubit=0)
test_circuit.apply_gate(Hadamard, qubit=1)
test_circuit.apply_gate(Hadamard, qubit=2)
test_circuit.apply_gate(Hadamard % Hadamard % Hadamard)
test_circuit.measure_states(qubit=0)
test_circuit.apply_gate(Hadamard, qubit=1)
print(test_circuit.qubit_array[0])
test_circuit.apply_gate(T_Gate, qubit=1)
test_circuit.apply_gate(Identity % T_Gate % Identity)
print(test_circuit.qubit_array[0])
test_circuit.measure_states(qubit=1)
test_circuit.measure_states(qubit=2)
test_circuit.list_probs(qubit=0)
test_circuit.list_probs(qubit=1)
test_circuit.list_probs(qubit=2)
print(test_circuit[0][0])
print(test_circuit[0][1])
print(test_circuit[0][2])
test_circuit.get_info()
large_circuit = Circuit(q=6)
large_circuit.apply_gate(Hadamard, qubit=0)
large_circuit.apply_gate(Hadamard, qubit=3)
large_circuit.apply_gate(Hadamard, qubit=4)

large_circuit.apply_gate(Hadamard, qubit=2)
large_circuit.apply_gate(CNot % CNot % T_Gate % Hadamard)
large_circuit.list_probs()
large_circuit.get_info()
large_qub = q0 % q0 % q0 % q0 % q0 % q0
other_large_qub = large_circuit.return_state()
slightly_smaller_qub = q0 % q0 % qp % q0 % qm
print(QuantInfo.quantum_mutual_info(slightly_smaller_qub, slightly_smaller_qub))
print(QuantInfo.fidelity(other_large_qub, other_large_qub))
print(QuantInfo.trace_distance(other_large_qub, other_large_qub))
print(QuantInfo.trace_distance_bound(other_large_qub, other_large_qub))

print(f"Cond ent:{QuantInfo.quantum_conditional_entropy(other_large_qub, other_large_qub)}")
print(f"Rel ent:{QuantInfo.quantum_relative_entropy(other_large_qub, other_large_qub)}")

print(type(other_large_qub.rho))
print(type(large_qub.rho))
print(QuantInfo.fidelity(large_qub, other_large_qub))
print(QuantInfo.trace_distance(large_qub, other_large_qub))
print(QuantInfo.trace_distance_bound(large_qub, other_large_qub))

print(type(other_large_qub.rho))
print(type(large_qub.rho))
print(large_qub)
print(other_large_qub @ large_qub)
print(type(other_large_qub @ large_qub))
even_smaller_qub = q0 % q0
print(f"Cond ent:{QuantInfo.quantum_conditional_entropy(even_smaller_qub, even_smaller_qub)}")
print(f"Rel ent:{QuantInfo.quantum_relative_entropy(even_smaller_qub, even_smaller_qub)}")
print(f"Discord:{QuantInfo.quantum_discord(even_smaller_qub, even_smaller_qub)}")
print(f"Mutual Info:{QuantInfo.quantum_mutual_info(even_smaller_qub, even_smaller_qub)}")
even_smaller_qub = q0 % q0 % qm
print(f"Cond ent:{QuantInfo.quantum_conditional_entropy(even_smaller_qub, even_smaller_qub)}")
print(f"Rel ent:{QuantInfo.quantum_relative_entropy(even_smaller_qub, even_smaller_qub)}")
print(f"Discord:{QuantInfo.quantum_discord(even_smaller_qub, even_smaller_qub)}")
print(f"Mutual Info:{QuantInfo.quantum_mutual_info(even_smaller_qub, even_smaller_qub)}")
even_smaller_qub = q0 % q0 % qm % qp
print(f"Cond ent:{QuantInfo.quantum_conditional_entropy(even_smaller_qub, even_smaller_qub)}")
print(f"Rel ent:{QuantInfo.quantum_relative_entropy(even_smaller_qub, even_smaller_qub)}")
print(f"Discord:{QuantInfo.quantum_discord(even_smaller_qub, even_smaller_qub)}")
print(f"Mutual Info:{QuantInfo.quantum_mutual_info(even_smaller_qub, even_smaller_qub)}")

hugeeee_qubit = q0 % q0
hugeeee_qubit_2 = qm % qm
hugeeee_gate = X_Gate % X_Gate
hugeeee_gate_2 = Hadamard % Hadamard
for i in range(6):
    hugeeee_qubit %= q0
    hugeeee_qubit_2 %= qm
    hugeeee_qubit @= hugeeee_qubit_2
    hugeeee_gate %= X_Gate
    hugeeee_gate_2 %= Hadamard
    hugeeee_gate @= hugeeee_gate_2
    print(f"Qubits {i}")
    print(type(hugeeee_qubit.rho))
    print(count_zeros(hugeeee_qubit.rho))
    print(type(hugeeee_qubit_2.rho))
    print(count_zeros(hugeeee_qubit_2.rho))
    print(f"Gates {i}")
    print(type(hugeeee_gate.matrix))
    print(count_zeros(hugeeee_gate.matrix))
    print(hugeeee_gate.matrix.size)
    print(type(hugeeee_gate_2.matrix))
    print(count_zeros(hugeeee_gate_2.matrix))
print(Hadamard @ Hadamard.matrix)
print(Hadamard @ q0_lw)
had = Hadamard @ Hadamard
for i in range(20):
    had @= Hadamard
print(q0_symb + q1_symb)
print(q0_symb + qgen_symb)
print(qgen_symb + qgen_symb)
print(q0_symb - q1_symb)
print(q0_symb - qgen_symb)
print(qgen_symb - qgen_symb)
print(q0_symb % q1_symb)
print(q0_symb % qgen_symb)
print(qgen_symb % qgen_symb)
print(q0_symb @ q1_symb)
print(q0_symb @ qgen_symb)
print(qgen_symb @ qgen_symb)
super_sparse_qub = Qubit.q0(n=4) % qp % qp % qp % qp % qp % qp
test = super_sparse_qub
print(type(test.rho))
print(count_zeros(test.rho))
print(test.rho.count_nonzero())
print(test.rho.size)
print(test.rho)
test_gate = (Gate.Identity(n=4) % Hadamard % Hadamard % Hadamard % Hadamard % Hadamard % Hadamard)
print(test_gate @ super_sparse_qub)
print((Identity % Hadamard) @ (q0 % qp))
print(Identity)
print(Hadamard)
print(q0.skip_val)
print(q0)
print(qp)
other_sparse_test = Qubit.q0(n=5)
sparse_test = Qubit.q0(n=9)
print(sparse_test[0])
print(sparse_test[4])
print(sparse_test @ sparse_test)
print(other_sparse_test % other_sparse_test)
print(test_gate @ test_gate)