from ..circuit.classes import *


test_circuit = Circuit(q=12)
for i in range(12):
    test_circuit.add_gate(Hadamard, i)
test_circuit.list_probs()





