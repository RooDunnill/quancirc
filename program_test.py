from qcp_program import *
rho_ab = Density(state=q1 @ q0 @ q0).rho
partial_trace = Density(rho=rho_ab).partial_trace(trace_out="B",state_size=2)
print_array(partial_trace)
partial_trace2 = Density(rho=rho_ab).partial_trace(trace_out="A",state_size=2)
print_array(partial_trace2)

rho_ab = Density(state=q1 @ q0).rho
partial_trace = Density(rho=rho_ab).partial_trace(trace_out="B",state_size=1)
print_array(partial_trace)
partial_trace2 = Density(rho=rho_ab).partial_trace(trace_out="A",state_size=1)
print_array(partial_trace2)

rho_ab = Density(state=q1 @ q0 @ q0).rho
partial_trace = Density(rho=rho_ab).partial_trace(trace_out="B",state_size=1)
print_array(partial_trace)
partial_trace2 = Density(rho=rho_ab).partial_trace(trace_out="A",state_size=1)
print_array(partial_trace2)

q00 = q0 @ q0
q11 = q1 @ q1
test_den_object = Density(state_a=q00, state_b=q11, state=q00 @ q11)
print_array(test_den_object.rho)
print_array(test_den_object.rho_a)
print_array(test_den_object.rho_b)
print_array(f"Rho B after tracing out A\n{test_den_object.partial_trace(trace_out="A", state_size=2)}")
print_array(f"Rho A after tracing out B\n{test_den_object.partial_trace(trace_out="B", state_size=2)}")
print_array(f"Trace Distance between A and B: {test_den_object.trace_distance()}")
print_array(f"Fidelity between state A and B: {test_den_object.fidelity()}")
print_array(f"Quantum Mutual Information S(A:B): {test_den_object.quantum_mutual_info()}")
print_array(f"Quantum Conditional Entropy S(A|B): {test_den_object.quantum_conditional_entropy(rho="A")}")
print_array(f"Quantum Conditional Entropy S(B|A): {test_den_object.quantum_conditional_entropy(rho="B")}")
print_array(f"Quantum Relative Entropy S(A||B): {test_den_object.quantum_relative_entropy(rho="A")}")
print_array(f"Quantum Relative Entropy S(B||A): {test_den_object.quantum_relative_entropy(rho="B")}")

qpp = qp @ qp
qmm = qm @ qm
test_den_object = Density(state_a=qpp, state_b=qmm, state=qpp @ qmm)
print_array(test_den_object.rho)
print_array(test_den_object.rho_a)
print_array(test_den_object.rho_b)
print_array(f"Rho B after tracing out A\n{test_den_object.partial_trace(trace_out="A", state_size=2)}")
print_array(f"Rho A after tracing out B\n{test_den_object.partial_trace(trace_out="B", state_size=2)}")
print_array(f"Trace Distance between A and B: {test_den_object.trace_distance()}")
print_array(f"Fidelity between state A and B: {test_den_object.fidelity()}")
print_array(f"Quantum Mutual Information S(A:B): {test_den_object.quantum_mutual_info()}")
print_array(f"Quantum Conditional Entropy S(A|B): {test_den_object.quantum_conditional_entropy(rho="A")}")
print_array(f"Quantum Conditional Entropy S(B|A): {test_den_object.quantum_conditional_entropy(rho="B")}")
print_array(f"Quantum Relative Entropy S(A||B): {test_den_object.quantum_relative_entropy(rho="A")}")
print_array(f"Quantum Relative Entropy S(B||A): {test_den_object.quantum_relative_entropy(rho="B")}")

test_circuit = Circuit(n=3)
test_circuit.add_single_gate(gate=Hadamard, gate_location=0)
print_array(Hadamard @ Identity @ Identity)
test_circuit.add_single_gate(gate=X_Gate, gate_location=1)
print_array(Identity @ X_Gate @ Identity)
test_circuit.run()
print_array(test_circuit.return_info("final_gate"))
Grover(oracle_values).run()
Grover(oracle_values2).run()
Grover(oracle_values3).run()
Grover(oracle_values4).run()
print_array(q0 @ q0 @ q0 @ q0)
print_array(Hadamard * q0)
print_array(type(Hadamard * q0))
print_array(Hadamard * Hadamard)
print_array(Qubit(type="seperable", vectors=[q0,q0,q1]))
print_array(Qubit(type="seperable", vectors=[[1,0],[1,0],[0,1]]))
print_array((Hadamard @ Hadamard) * Qubit(type="seperable", vectors=[q0,q0]))
had_mult_test = Qubit(type="seperable", vectors=[q0,q0])
print_array(had_mult_test)
print_array((Hadamard @ Hadamard) * had_mult_test)
oracle_values = [9,4,3,2,5,6,12,15,16]
oracle_values2 = [1,2,3,4,664,77,5,10,12,14,16,333,334,335,400,401,41,42,1000]
oracle_values3 = [4,5,30,41]
oracle_values4 = [500,5,4,7,8,9,99]
oracle_value_test = [1,2,3]
large_oracle_values = [1120,2005,3003,4010,5000,6047,7023,8067,9098,10000,11089,12090,13074]
Grover(16, fast=True, iter_calc="round").run()
Grover(16, fast=True, iter_calc="floor").run()
Grover(16, fast=True, iter_calc="balanced").run()
Grover(oracle_values2, fast=True, iter_calc="round").run()
Grover(oracle_values2, fast=True, iter_calc="floor").run()
Grover(oracle_values2, fast=True, iter_calc="balanced").run()
Grover(large_oracle_values, fast=True, iter_calc="round").run()
Grover(large_oracle_values, fast=True, iter_calc="floor").run()
Grover(large_oracle_values, fast=True, iter_calc="balanced").run()