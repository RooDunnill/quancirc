from classes import *


noisy_circuit = Circuit(q=2)
noisy_circuit.apply_channel_to_qubit(0, "X", 0.5)
noisy_circuit.state.set_display_mode("both")
noisy_circuit.state.debug()
print(noisy_circuit.state.skip_val)
print(noisy_circuit.state)
noisy_circuit.apply_channel_to_qubit(1, "X", 0.5)
print(noisy_circuit.state)

"""
noisy_local = Circuit(q=2)
noisy_local.apply_local_channel_to_qubit("X", 0.5)
noisy_local.state.set_display_mode("both")
noisy_local.state.debug()
print(noisy_local.state.skip_val)
print(noisy_local.state)

"""