from ..circuit.classes import *



noisy_circuit = Circuit(q=2)
noisy_circuit.apply_channel_to_qubit(0, "X", 0.5)
noisy_circuit.state.set_display_mode("both")
noisy_circuit.state.debug()
print(noisy_circuit.state.skip_val)
print(noisy_circuit.state.rho)
print(QuantInfo.purity(noisy_circuit.state))
print(noisy_circuit.state.state_type)
print(noisy_circuit.state)
