import sys; sys.path.append("..")
from classes import *
from utilities import *

partial_test = q0 @ q1 @ q1 @ q0
pt = partial_test.partial_trace(trace_out="B", state_size=3)
pt2 = partial_test.partial_trace(trace_out="A", state_size=3)
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