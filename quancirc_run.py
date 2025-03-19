from classes import *




test_circuit = Circuit(q=2)
test_circuit.add_gate(Identity, qubit=0)
test_circuit.add_gate(Hadamard, qubit=1)
test_circuit.apply_gates()
test_circuit.list_probs()
test_circuit.measure_state()
print(test_circuit)

test_circuit = Circuit(q=2)
test_circuit.add_gate(Identity, 0)
test_circuit.add_gate(Hadamard, 1)
test_circuit.apply_gates()
test_circuit.list_probs()
test_circuit.measure_state(1)
print(test_circuit)