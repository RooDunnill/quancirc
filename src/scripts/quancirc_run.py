from ..circuit.classes import *

test_qub = q0 % qm % qp
eigvals, eigvecs = np.linalg.eigh(test_qub.rho)
diag_rho = np.diag(eigvals)
print(diag_rho)
diag_qub = Qubit(rho=diag_rho)
print(eigvecs)
print(diag_qub.partial_trace_gen_gen(2,0))
print(diag_qub.partial_trace_gen_gen(1,1))


test_circuit = Circuit(q=5)
for i in range(5):
    test_circuit.add_gate(Hadamard, i)
test_circuit.list_probs()
print(Hadamard @ qp)
print(X_Gate @ q0)


test_qub = q0 % qp % qp
test_qub.set_display_mode("density")
print(test_qub.partial_trace("B", 2))
print(test_qub.partial_trace("A",2))
print(test_qub.partial_trace("A",1))

test_qub = q0 % qp % qp
test_qub.set_display_mode("density")
print(test_qub.partial_trace_gen("B", 2))
print("test")
print(test_qub.partial_trace_gen("A",2))
print("test")
print(test_qub.partial_trace_gen("A",1))


test_qub = qm % qp % qp % q0
test_qub.set_display_mode("density")
print(test_qub.partial_trace_gen_gen(0,2))
print("test")
print(test_qub.partial_trace_gen_gen(2,0))
print("test")
print(test_qub.partial_trace_gen_gen(1, 1))