import sys; sys.path.append("..")
from classes import *
from utilities import *

partial_test = q0 @ q1 @ q1 @ q0
pt = partial_test.partial_trace("B", 3)
pt2 = partial_test.partial_trace("A", 3)
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
partial_test = q0 @ q1 @ q0
print(partial_test.decompose_state(0))
print(partial_test.decompose_state(1))
print(partial_test.decompose_state(2))
print(f"\n\n\n\n\n\n")
set_test = q1 @ q1 @ q1
set_test.debug()
set_test[0] = Hadamard * q0
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
state_1 = q0 @ q0
state_2 = q1 @ q1

QuantInfo.two_state_info(state_1, state_2)
print(Hadamard)
print(Hadamard @ X_Gate)
print(q0)
print(q0 @ q0)
q0.set_display_mode("both")
print(q0)

and_test1 = Bit(4, val=0)
and_test2 = Bit(4, val=1)
and_test2[1] = 0
and_test1[2] = 1
print(and_test1 & and_test2)
print(and_test1 | and_test2)
print(and_test1 ^ and_test2)
measure_test_qubit = qm @ qp
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
test_state = qm @ qm
test_state.set_display_mode("both")
povm = [[[1,0],[0,0]],[[0,0],[0,1]]]
print(test_state.display_mode)
pm_qub = Measure(test_state).measure_state()
print(pm_qub.display_mode)
print(pm_qub)
print(Measure(test_state[0]).measure_state(povm))

display_qub = q0 @ q0
print(display_qub)
print(display_qub.display_mode)
display_qub.set_display_mode("both")
print(display_qub)
pm_qub.debug()