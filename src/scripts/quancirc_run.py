from ..circuit.classes import *



circuit_test = Circuit(q=4)
circuit_test.add_gate(Hadamard, 2)
print(circuit_test.state)
circuit_test.add_gate(Hadamard, 3)
print(circuit_test[2])
circuit_test.list_probs()
