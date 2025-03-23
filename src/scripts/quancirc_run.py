from ..circuit.classes import *



circuit_test = Circuit(q=4)
circuit_test.add_gate(Hadamard, 2)
print(circuit_test.state)
circuit_test.add_gate(Hadamard, 3)
print(circuit_test[2])
circuit_test.list_probs()


parital_qub = q1 % q0 % q0 % q1 
parital_qub.set_display_mode("density")
new_qub = parital_qub.partial_trace_gen(1,1)
print(new_qub)
print("X" * 10)
new_qub = parital_qub.partial_trace_gen(2,0)
print(new_qub)
print("X" * 10)
new_qub = parital_qub.partial_trace_gen(0,2)
print(new_qub)