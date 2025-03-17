import sys; sys.path.append("..")
from classes import *
from utilities import *
print(Hadamard)
test = Qubit(state=[1,0])
hopefully_qplus = Hadamard * test
print(hopefully_qplus)

hopefully_qplus.set_display_mode("both")
print(hopefully_qplus)

print(Hadamard * Hadamard)
print(Hadamard @ Hadamard)
partial_test = q0 @ q1 @ q0
pt = partial_test.partial_trace(trace_out="B", state_size=1)
pt2 = partial_test.partial_trace(trace_out="A", state_size=1)
print(pt)
print(pt2)
partial_test = q0 @ q1 @ q0
pt = partial_test.partial_trace(trace_out="B", state_size=2)
pt2 = partial_test.partial_trace(trace_out="A", state_size=2)
print(pt)
print(pt2)