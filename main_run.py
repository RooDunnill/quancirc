import time
start = time.time()
import numpy as np                                              #mostly used to make 1D arrays
from random import choices                                            #used for measuring
import atexit
import matplotlib.pyplot as plt
from rich.console import Console
from rich.theme import Theme
from scipy.linalg import sqrtm, logm
import cProfile
from functools import lru_cache, cache
from qcp_program import *
console = Console()


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

oracle_vals = [9,4,3,2,5,6,12,15,16,17]
print(oracle_vals)
Grover(oracle_vals).run()


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