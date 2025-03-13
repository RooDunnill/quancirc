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
print_array(f"Quantum Conditional Entropy S(A|B): {test_den_object.quantum_conditional_entropy()}")
print_array(f"Quantum Conditional Entropy S(B|A): {test_den_object.quantum_conditional_entropy()}")
print_array(f"Quantum Relative Entropy S(A||B): {test_den_object.quantum_relative_entropy()}")
print_array(f"Quantum Relative Entropy S(B||A): {test_den_object.quantum_relative_entropy()}")

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
print_array(f"Quantum Conditional Entropy S(A|B): {test_den_object.quantum_conditional_entropy()}")
print_array(f"Quantum Conditional Entropy S(B|A): {test_den_object.quantum_conditional_entropy()}")
print_array(f"Quantum Relative Entropy S(A||B): {test_den_object.quantum_relative_entropy()}")
print_array(f"Quantum Relative Entropy S(B||A): {test_den_object.quantum_relative_entropy()}")

test_circuit = Circuit(n=3)
test_circuit.add_single_gate(gate=Hadamard, gate_location=0)
print_array(Hadamard @ Identity @ Identity)
test_circuit.add_single_gate(gate=X_Gate, gate_location=1)
print_array(Identity @ X_Gate @ Identity)
test_circuit.run()
print_array(test_circuit.get_info("final_gate"))
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
large_oracle_values = [1120,2005,3003,4010,5000,6047,7023,8067,9098,10000,11089,12090,13074]

povm1 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
povm2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
povm3 = np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
povm4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
qub = q0 @ qm
print_array(qub)
print_array(Measure(state=qub).list_probs(povm=[povm1,povm2,povm3,povm4]))
print_array(Measure(state=qub).measure_state(povm=[povm1,povm2,povm3,povm4]))
print_array(Measure(state=qub).list_probs())
test = Measure(state=qub).measure_state(text=True)
print_array(test)
vne_test = Qubit(type="seperable", vectors=[qp,qm,q1], detailed=True)
print_array(vne_test.density)
se_test = Qubit(type="mixed", vectors=[q0,q1], weights=[0.2,0.8],detailed=True)
print_array(se_test)
print_array(se_test.density)
print_array(se_test.se)
se_test = Qubit(type="mixed", vectors=[[1,0],[0,1]], weights=[0.2,0.8],detailed=True)
print_array(se_test)
print_array(se_test.density)
print_array(se_test.se)
povm1 = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
povm2 = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
povm3 = np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
povm4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
qub = q0 @ qm
print_array(qub)
print_array(Measure(state=qub).list_probs(povm=[povm1,povm2,povm3,povm4]))
print_array(Measure(state=qub).measure_state(povm=[povm1,povm2,povm3,povm4]))
print_array(Measure(state=qub).list_probs())
test = Measure(state=qub).measure_state(text=True)
print_array(test)
vne_test = Qubit(type="seperable", vectors=[qp,qm,q1], detailed=True)
print_array(vne_test.density)
se_test = Qubit(type="mixed", vectors=[q0,q1], weights=[0.2,0.8],detailed=True)
print_array(se_test)
print_array(se_test.density)
print_array(se_test.se)
se_test = Qubit(type="mixed", vectors=[[1,0],[0,1]], weights=[0.2,0.8],detailed=True)
print_array(se_test)
print_array(se_test.density)
print_array(se_test.se)
print_array(Q * q0)
rho_Alice = np.array([1/3,0,0,0,1/3,0,0,0,1/3])
rho_Bob = np.array([1/2,0,0,0,1/2,0,0,0,0])
trace_calc = Density(rho_a=rho_Alice, rho_b=rho_Bob)
qub = Qubit(type="seperable", vectors=[q0,q1,q0,q0])
print_array(Density(state=qub).partial_trace(trace_out="B", state_size=3))
prob_test = Measure(state=qub).list_probs(qubit=2)
measure_test = Measure(state=qub).list_probs()
print_array(f"This is the prob list of the function: \n{measure_test}")
print_array(f"This is the probs of the individual qubit: \n{prob_test[0]}")
prob_test = Measure(state=qub).measure_state(qubit=2)
print_array(prob_test)
print_array(Q * q0)
print(binary_entropy(0.125*0.5))
print(-1/8*np.log2(1/8) - (1 - 1/8)*np.log2(1 - 1/8))
print(np.log2(7/8))
print(3/8-7/8*np.log2(7/8))
qcs_coursework = Qubit(type="mixed", vectors=[q0, qp], weights=[0.5,0.5], detailed=True)
print_array(qcs_coursework.vne)
print_array(binary_entropy(1/4))
print_array(np.linalg.eig(reshape_matrix(np.array([0.75,0.25,0.25,0.25]))))
print_array(-0.853553*np.log2(0.853553)-0.146447*np.log2(0.146447))
print_array(Density(state_a = q1, state_b = qpi).fidelity())
Bell = Circuit(n=2)
Bell.add_single_gate(gate=Hadamard @ Hadamard, gate_location=0)
Bell.add_single_gate(gate=CNot, gate_location=0)
Bell.apply_final_gate()
Bell.list_probs()
Bell.print_gates()
print_array(Density(state_a = q1, state_b = qpi).fidelity())
example_circuit = Circuit(n=4)
example_circuit.add_single_gate(gate=Hadamard, gate_location=0)
example_circuit.add_gate(Z_Gate @ Hadamard @ Identity @ X_Gate)
example_circuit.add_gate(S_Gate @ S_Gate @ Hadamard @ Hadamard)
example_circuit.add_gate(Hadamard @ Hadamard @ Hadamard @ Hadamard)
example_circuit.add_single_gate(CNot, gate_location=0)
example_circuit.apply_final_gate()
example_circuit.list_probs()
example_circuit.measure_state(qubit=2)
single_qubit_measurement_test = Circuit(n=3)
single_qubit_measurement_test.add_single_gate(Hadamard, gate_location=1)
single_qubit_measurement_test.add_single_gate(Hadamard, gate_location=2)
single_qubit_measurement_test.apply_final_gate()
single_qubit_measurement_test.measure_state(qubit = 2)

noisy_circuit = Circuit(n=2, noisy=True, Q_channel="P flip", prob=0.3)
noisy_circuit.apply_final_gate()
noisy_circuit.add_quantum_channel(Q_channel="B flip", prob=0.4)
noisy_circuit.list_probs()


Q_channel_circuit = Circuit(n = 4)
Q_channel_circuit.add_gate(Hadamard @ Hadamard @ Hadamard @ Hadamard)

Q_channel_circuit.apply_final_gate()
Q_channel_circuit.add_quantum_channel(Q_channel="P flip", prob=0.8)
Q_channel_circuit.list_probs()
oracle_values = [9,4,3,2,5,6,12,15,16]
oracle_values2 = [1,2,3,4,664,77,5,10,12,14,16,333,334,335,400,401,41,42,1000]
oracle_values3 = [4,5,30,41]
oracle_values4 = [500,5,4,7,8,9,99]
oracle_value_test = [1,2,3]
Grover(oracle_values).run()
Grover(oracle_values2).run()
Grover(oracle_values3).run()
Grover(oracle_values4).run()
Grover(16, fast=True, iter_calc="round").run()
Grover(16, fast=True, iter_calc="floor").run()
Grover(16, fast=True, iter_calc="balanced").run()
Grover(oracle_values2, fast=True, iter_calc="round").run()
Grover(oracle_values2, fast=True, iter_calc="floor").run()
Grover(oracle_values2, fast=True, iter_calc="balanced").run()
Grover(large_oracle_values, fast=True, iter_calc="round").run()
Grover(large_oracle_values, fast=True, iter_calc="floor").run()
Grover(large_oracle_values, fast=True, iter_calc="balanced").run()
Grover(oracle_values, n=10, iterations=10).run()
Grover(oracle_values, n=10).run()
time_test(16)
comp_Grover_test(12)
demo_circuit = Circuit(n=4)
demo_circuit.add_gate(Hadamard @ Identity @ Hadamard @ Identity)
demo_circuit.add_gate(Identity @ X_Gate @ Identity @ X_Gate)
demo_circuit.add_single_gate(gate=Hadamard, gate_location=0)
demo_circuit.apply_final_gate()
demo_circuit.list_probs()
demo_circuit.measure_state()
demo1 = Circuit(n=4)
demo1.add_gate(Hadamard @ Hadamard @ Hadamard @ Hadamard)
demo1.add_gate(Identity @ Z_Gate @ Z_Gate @ S_Gate)
demo1.add_gate(CNot @ CNot)
demo1.add_single_gate(gate=Hadamard, gate_location=3)
demo1.apply_final_gate()
demo1.list_probs()
demo1.measure_state(qubit=2)
demo1.get_von_neumann()
demo1.get_von_neumann(qubit=0)
demo1.get_von_neumann(qubit=1)
demo1.get_von_neumann(qubit=2)
demo1.get_von_neumann(qubit=3)

demo2 = Circuit(n=2, noisy=True, Q_channel="B flip", prob=0.3)
demo2.add_single_gate(gate=Hadamard, gate_location=0)
demo2.apply_final_gate()
demo2.add_quantum_channel(Q_channel="P flip", prob=0.2)
demo2.add_gate(Hadamard @ X_Gate)
demo2.add_gate(S_Gate @ Hadamard)
demo2.add_gate(CNot)
demo2.apply_final_gate()
demo2.list_probs()
demo2.measure_state()

Bell = Circuit(n=2)
Bell.add_single_gate(gate=Hadamard, gate_location=0)
Bell.add_single_gate(gate=CNot, gate_location=0)
Bell.add_gate(Hadamard @ X_Gate)
Bell.apply_final_gate()
Bell.add_quantum_channel(Q_channel="B P flip", prob = 0.4)
Bell.add_quantum_channel(Q_channel="B flip", prob = 0.5)
Bell.list_probs()
Bell.measure_state(qubit=0)
Bell.get_von_neumann(qubit=0)
Grover(1, fast=True, n=14).run()
print_array(Density(state=qpi))
print_array(Density(state=qmi))

qcs = Circuit(n=1, state=qp, noisy=True, Q_channel = "P flip", prob=0.9)
qcs.get_info("state")
qcs.apply_final_gate()
qcs.list_probs()
print_array(Hadamard + Hadamard)
print_array(Hadamard - Hadamard)
testp = Density(state=qp)
testm = Density(state = qm)
print_array(testp + testm)
print_array(testp - testm)
