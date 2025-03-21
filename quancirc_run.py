from classes import *


noisy_circuit = Circuit(q=2)
noisy_circuit.add_quantum_channel("X", 0.5)
noisy_circuit.state.set_display_mode("both")
print(noisy_circuit.state)