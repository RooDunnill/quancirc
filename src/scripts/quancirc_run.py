from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *


print(q0_lw)
print(q0_lw % q0_lw)
lw_circuit = Circuit_LW(q=4)
lw_circuit.add_gate(Hadamard, qubit=1)
print(type(lw_circuit.state))
lw_circuit.add_gate(Hadamard, qubit=2)
lw_circuit.add_gate(Identity % Hadamard % Hadamard % Identity)
lw_circuit.list_probs()
test = q0_lw % q0_lw
print(test)
print(type(test.rho))
