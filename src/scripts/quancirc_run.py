from ..circuit.classes import *




test_qub = q0 % qm

print((Hadamard % Hadamard) @ test_qub)
print(Gate.Identity(n=1) % Gate.Identity(n=1))
print(Identity % Identity)
test_circuit = Circuit(q=2)
test_circuit.add_gate(Identity, qubit=0)
test_circuit.add_gate(Hadamard, qubit=1)

test_circuit.list_probs()
test_circuit.measure_state()
print(test_circuit)