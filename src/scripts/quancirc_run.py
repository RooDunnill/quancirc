from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *


print(q0_lw)
print(q0_lw % q0_lw)
lw_circuit = Circuit_LW(q=4)
lw_circuit.add_gate(Hadamard, qubit=1)
lw_circuit.add_gate(Hadamard, qubit=2)
