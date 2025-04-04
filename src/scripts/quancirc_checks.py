#!/usr/bin/env python
import numpy as np
from ..circuits import *
from ..circuit_algorithms.grover_search import *
import sympy as sp


partial_test = q0 % q1 % q1 % q0
pt = partial_test.partial_trace(0, 3)
pt2 = partial_test.partial_trace(3, 0)
print(pt)
print(pt2)



print(partial_test.isolate_qubit(0))
print(partial_test.isolate_qubit(1))
print(partial_test.isolate_qubit(2))
print(partial_test.isolate_qubit(3))
print(partial_test[0])
print(partial_test[1])
print(partial_test[2])
print(partial_test[3])
qubit_list = partial_test[:]
print(qubit_list)
print(qubit_list[0])
print(qubit_list[1])
print(qubit_list[2])
print(qubit_list[3])
partial_test = q0 % q1 % q0
print(partial_test.decompose_state(0))
print(partial_test.decompose_state(1))
print(partial_test.decompose_state(2))
print(f"\n\n\n\n\n\n")
set_test = q1 % q1 % q1
set_test.debug()
print(set_test)
print(type(set_test.rho))
set_test[0] = Hadamard @ q0
set_test.debug()
print(Qubit.q0(n=2))
fwht_state = Qubit.q0(n=4)
Quan_qub_1 = q0
Quan_qub_2 = q1
print(QuantInfo.fidelity(Quan_qub_1, Quan_qub_2))
print(QuantInfo.trace_distance(Quan_qub_1, Quan_qub_2))

QuantInfo.two_state_info(Quan_qub_1, Quan_qub_2)

state_1 = q0 % q0
state_2 = q1 % q1

QuantInfo.two_state_info(state_1, state_2)
print(Hadamard)
print(Hadamard % X_Gate)
print(q0)
print(q0 % q0)
q0.set_display_mode("both")
print(q0)


measure_test_qubit = qm % qp
measure_test = Measure(state=measure_test_qubit).list_probs()
print(measure_test)
pm_state = Measure(state=measure_test_qubit).measure_state()
print(type(pm_state))
pm_state.set_display_mode("both")
print(pm_state)
print(pm_state.display_mode)
measure_test_qub1 = Measure(state = measure_test_qubit[1]).list_probs()
print(measure_test_qub1)
pm_state_qub1 = Measure(state = measure_test_qubit[1]).measure_state()
pm_state_qub1.set_display_mode("both")
print(pm_state_qub1)
measure_test_qub0 = Measure(state = measure_test_qubit[0]).list_probs()
print(measure_test_qub0)
pm_state_qub0 = Measure(state = measure_test_qubit[0]).measure_state()
pm_state_qub0.set_display_mode("both")
print(pm_state_qub0)
povm = [[[1,0],[0,0]],[[0,0],[0,1]]]
povm_qub = q0
print(Measure(state=povm_qub).list_probs(povm=povm))
print(Measure(state=povm_qub).measure_state(povm=povm))
test_state = qm % qm
test_state.set_display_mode("both")
povm = [[[1,0],[0,0]],[[0,0],[0,1]]]
print(test_state.display_mode)
pm_qub = Measure(test_state).measure_state()
print(pm_qub.display_mode)
print(pm_qub)
print(Measure(test_state[0]).measure_state(povm))

display_qub = q0 % q0
print(display_qub)
print(display_qub.display_mode)
display_qub.set_display_mode("both")
print(display_qub)
pm_qub.debug()
test_circuit = Circuit(states=1,q=2)
print("\n")
print(f"QUBIT ARRAY:\n{test_circuit.qubit_array}")
test_circuit.apply_gate(Identity, qubit=0)
test_circuit.apply_gate(Hadamard, qubit=1)

test_circuit.list_probs()
test_circuit.measure_state()
print(test_circuit)

test_circuit = Circuit(states=1, q=2)
test_circuit.apply_gate(Identity, qubit=0)
test_circuit.apply_gate(Hadamard, qubit=1)

test_circuit.list_probs()
test_circuit.measure_state(qubit=1)
print(test_circuit)

attr_test_0 = q0
attr_test_0.set_display_mode("both")
attr_test_1 = q1
attr_test_1.set_display_mode("density")
attr_sum = attr_test_0 % attr_test_1
print(attr_sum.display_mode)
print(attr_sum.skip_val)
print(type(attr_sum))
print(attr_sum)
index_test = q0 % q0 % q0
test = index_test[0]
print(test.index)
test2 = test % q0
print(test2.index)

test_qub = q0 * 2
print(test_qub)
print(test_qub.skip_val)

noisy_circuit = Circuit(states=1, q=2)
noisy_circuit.apply_channel_to_qubit(0, 0, "X", 0.5)
noisy_circuit.qubit_array[0].set_display_mode("density")
noisy_circuit.qubit_array[0].debug()
print(noisy_circuit.qubit_array[0].skip_val)
print(noisy_circuit.qubit_array[0])
noisy_circuit.apply_channel_to_qubit(0, 1, "X", 0.5)
print(noisy_circuit.qubit_array[0])


noisy_local = Circuit(states=1, q=2)
noisy_local.apply_local_channel_to_state(0, "X", 0.5)
noisy_local.qubit_array[0].set_display_mode("density")
noisy_local.debug()
print(noisy_local.qubit_array[0].skip_val)
print(noisy_local.qubit_array[0])
info_test = Circuit(states=1, q=4)
info_test.get_info()
info_test.linear_entropy()
info_test.purity()
info_test.vn_entropy()
info_test.shannon_entropy()
qub_0 = q1 % qp
qub_p = q1 % qp
qub_p.set_display_mode("both")
print(qub_p)
print(QuantInfo.trace_distance_bound(qub_0, qub_p))
print(QuantInfo.trace_distance(qub_0, qub_p))
QuantInfo.two_state_info(qub_0, qub_p)
print(Gate.Rotation_X(np.pi))
print(Gate.Rotation_Y(np.pi))
print(Gate.Rotation_Z(np.pi))

print(Hadamard @ qp)
print(X_Gate @ q0)
test_qub = Qubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
test_qub.set_display_mode("density")
test_qub.debug()
print(test_qub)

test_qub2 = test_qub % q0
print(test_qub2)
print(test_qub2.partial_trace(1,0))
print(test_qub2.partial_trace(0,1))
test_circuit = Circuit(states=1, q=8)
for i in range(8):
    test_circuit.apply_gate(Hadamard, qubit=i)
test_circuit.list_probs()

test_qub = qm % qp % qp % qm %qm
test_qub.partial_trace(2,2)
test_qub.partial_trace(1,3)
print(test_qub.partial_trace(0,1))
print(test_qub.partial_trace(0,4))
fwht_test =Circuit(q=8)

fwht_test.list_probs()
fwht_test.qubit_array[0].set_display_mode("density")
print(fwht_test.qubit_array[0])


print(Gate.Identity(n=4) % Gate.Identity(n=4))
sparse_test = q0 % q0
print(type(sparse_test.rho))

print(sparse_test)
print(Hadamard @ q1)
print(Hadamard % Hadamard)
print(Hadamard % Hadamard % Hadamard)
print(q0 % q0 % q0)
print(Hadamard @ q0)
print(Hadamard @ Hadamard @ Hadamard)
print((Hadamard % Identity) @ (q0 % q0))
test_qub = q0 % qm

print((Hadamard % Hadamard) @ test_qub)
print(Gate.Identity(n=1) % Gate.Identity(n=1))
print(Identity % Identity)
test_circuit = Circuit(q=2)
test_circuit.apply_gate(Identity, qubit=0)
test_circuit.apply_gate(Hadamard, qubit=1)

test_circuit.list_probs()
test_circuit.measure_state()
print(test_circuit)
qubs = 6
test_circuit = Circuit(q=qubs)
for i in range(qubs):
    test_circuit.apply_gate(Hadamard, qubit=i)
    print(type(test_circuit.qubit_array[0].rho))
test_circuit.list_probs()

qub_partial = qm % qm % qm % qp % q0
print(qub_partial.partial_trace(1,0))
print(qub_partial.partial_trace(1,1))
print(qub_partial.partial_trace(1,2))
print(qub_partial.partial_trace(3,0))
print(qub_partial.partial_trace(0,3))
print(qub_partial.partial_trace(2,1))
print(qub_partial.partial_trace(0,1))

quant_info_test = q0 % qp % qm % q1
QuantInfo.state_info(quant_info_test)
print(Hadamard @ q0_lw)

print(q0_lw)
print(q0_lw % q0_lw)
lw_circuit = LwCircuit(q=4)
lw_circuit.apply_gate(Hadamard, qubit=1)
print(type(lw_circuit.state))
lw_circuit.apply_gate(Hadamard, qubit=2)
lw_circuit.apply_gate(Identity % Hadamard % Hadamard % Identity)
lw_circuit.list_probs()
test = q0_lw % q0_lw
print(test)
print(test.state)
grover_search(16, n=16, verbose=False)
test = grover_search(16, verbose=False)
print(test)
test_qubit_array = QubitArray(name="test")
print(len(test_qubit_array))
test_qubit_array.add_state(q0)
test_qubit_array.add_state(q1)
test_qubit_array.add_state(qm)
test_qubit_array.add_state(qp)
print(len(test_qubit_array))
test_qubit_array.qubit_info(1)
test_qubit_array.pop_first_state()
print(len(test_qubit_array))
test_qubit_array.qubit_info(0)
test_qubit_array.insert_state(qpi, 0)
test_qubit_array.qubit_info(1)
print(len(test_qubit_array))
test_qubit_array.validate_array()
qubit_array_circuit = Circuit()
qubit_array_circuit.upload_qubit_array(test_qubit_array)
qubit_array_circuit.apply_gate(Hadamard, index=2, qubit=0)
test_qubit_array = qubit_array_circuit.download_qubit_array()
bit_test = Bit("00001010")
print(bit_test)
bit_test_2 = Bit("01010101")
xor_bits = bit_test ^ bit_test_2
print(xor_bits)
xor_bits[0] = 1
print(xor_bits)
print(xor_bits ^ bit_test)
test = Gate.Hadamard
qub = Qubit.q0()
qub2 = Qubit(state=[1,0])
print(qub.__dir__())
print(dir(qub))
print(qub2.__dir__())
print(dir(qub2))
test_gate = Gate(matrix=[[1,0],[0,1]])

trace_state_1 = Qubit(state=[[1,0],[0,1]], weights=[0.5,0.5])

print(QuantInfo.trace_distance(trace_state_1, Gate.Rotation_Y(np.pi/2) @ trace_state_1))
print(QuantInfo.trace_distance(trace_state_1, Gate.Rotation_Y(np.pi/2) @ trace_state_1))

state_1 = Qubit(state=[1,0])

phi = sp.symbols("phi", real=False)
a, b, c = sp.symbols("a b c", real=False)
d, e, f = sp.symbols("d e f", real=False)
state_1 = SymbQubit(rho=sp.Matrix([[a, b], [sp.conjugate(b), c]]),skip_validation=True)
state_2 = SymbQubit(rho=sp.Matrix([[d, e], [sp.conjugate(e), f]]),skip_validation=True)
state_1.rho = state_1.rho.subs({b:0.0, a:sp.cos(phi)**2, c:sp.sin(phi)**2,})
state_2.rho = state_2.rho.subs({d:sp.sin(phi)**2, f:sp.cos(phi)**2, e:0.0})

expression = SymbQuantInfo.trace_distance(state_1, state_2)
print(expression.subs({phi:0.0}))

expression = SymbQuantInfo.fidelity(state_1, state_2)
print(expression.subs({phi:0.0}))

expression = SymbQuantInfo.trace_distance_bound(state_1, state_2)
print(expression[0].subs({phi:0.0}))
print(expression[1].subs({phi:0.0}))

print(q0_symb)
test_state = SymbQubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
print(test_state.state)
print(test_state)
print(test_state.state[2])
print(type(test_state.state[0]))
print(type(test_state.state))
print(Hadamard_symb @ test_state)
print(Hadamard_symb @ q0_symb)
print(P_Gate_symb @ qp_symb)
test_mixed_state = Qubit.create_mixed_state([q0,q1],[0.5,0.5])
print(test_mixed_state)

test = SymbQubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
print(test)

test2 = SymbQubit(state=[[1,0],[0,1],[sp.sqrt(0.5),sp.sqrt(0.5)]], weights=[0.2,0.2,0.6])
print(test2)

test3 = SymbQubit(state=[[1,0],[0,1],[sp.sqrt(0.5),sp.sqrt(0.5)]], weights=[sp.symbols("alpha"),0.2,0.6])
print(test3)
