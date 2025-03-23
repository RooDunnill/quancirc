from ..circuit.classes import *



circuit_test = Circuit(q=3)
circuit_test.add_gate(X_Gate, 0)
circuit_test.add_gate(Hadamard, 1)
circuit_test.state.set_display_mode("density")
print(circuit_test.state)
circuit_test.add_gate(Hadamard, 2)
print(circuit_test[2])
circuit_test.list_probs()
