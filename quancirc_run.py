from classes import *


print(Hadamard @ Identity)
print(CNot)
print(CNot * (Hadamard @ Identity))

Bell = Circuit(q=2)
Bell.add_gate(Hadamard, 0)
Bell.add_gate(CNot)
Bell.apply_gates()
Bell.measure_state()
print(Bell.state)
Bell.measure_state()
print(Bell.state)
print(Bell.purity())