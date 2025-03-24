#!/usr/bin/env python
from ..circuit.classes import *

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
bit_test = Bit(4)
bit_test[2] = True
print(bit_test)
state_1 = q0 % q0
state_2 = q1 % q1

QuantInfo.two_state_info(state_1, state_2)
print(Hadamard)
print(Hadamard % X_Gate)
print(q0)
print(q0 % q0)
q0.set_display_mode("both")
print(q0)

and_test1 = Bit(4, val=0)
and_test2 = Bit(4, val=1)
and_test2[1] = 0
and_test1[2] = 1
print(and_test1 & and_test2)
print(and_test1 | and_test2)
print(and_test1 ^ and_test2)
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
test_circuit = Circuit(q=2)
test_circuit.add_gate(Identity, qubit=0)
test_circuit.add_gate(Hadamard, qubit=1)

test_circuit.list_probs()
test_circuit.measure_state()
print(test_circuit)

test_circuit = Circuit(q=2)
test_circuit.add_gate(Identity, qubit=0)
test_circuit.add_gate(Hadamard, qubit=1)

test_circuit.list_probs()
test_circuit.measure_state(qubit=1)
print(test_circuit)

attr_test_0 = q0
attr_test_0.set_display_mode("both")
attr_test_0.skip_val = True
print(attr_test_0.skip_val)
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

q0.skip_val = True
test_qub = q0 * 2
print(test_qub)
print(test_qub.skip_val)

noisy_circuit = Circuit(q=2)
noisy_circuit.apply_channel_to_qubit(0, "X", 0.5)
noisy_circuit.state.set_display_mode("both")
noisy_circuit.state.debug()
print(noisy_circuit.state.skip_val)
print(noisy_circuit.state)
noisy_circuit.apply_channel_to_qubit(1, "X", 0.5)
print(noisy_circuit.state)


noisy_local = Circuit(q=2)
noisy_local.apply_local_channel_to_state("X", 0.5)
noisy_local.state.set_display_mode("both")
noisy_local.debug()
print(noisy_local.state.skip_val)
print(noisy_local.state)
info_test = Circuit(q=4)
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
test_circuit = Circuit(q=8)
for i in range(8):
    test_circuit.add_gate(Hadamard, i)
test_circuit.list_probs()

test_qub = qm % qp % qp % qm %qm
test_qub.partial_trace(2,2)
test_qub.partial_trace(1,3)
print(test_qub.partial_trace(0,1))
print(test_qub.partial_trace(0,4))
fwht_test =Circuit(q=8)
fwht_test.add_gate(Hadamard)
fwht_test.list_probs()
fwht_test.state.set_display_mode("density")
print(fwht_test.state)


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