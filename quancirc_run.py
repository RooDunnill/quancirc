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