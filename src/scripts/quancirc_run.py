from ..circuit.classes import *


qubs = 6
test_circuit = Circuit(q=qubs)
for i in range(qubs):
    test_circuit.add_gate(Hadamard, i)
    print(type(test_circuit.state.rho))
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