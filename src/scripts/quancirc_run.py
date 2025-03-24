from ..circuit.classes import *






fwht_test =Circuit(q=8)
fwht_test.add_gate(Hadamard)
fwht_test.list_probs()
fwht_test.state.set_display_mode("density")
print(fwht_test.state)


print(Gate.Identity(n=6) % Gate.Identity(n=6))



