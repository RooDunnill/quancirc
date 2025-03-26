from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *






bit_test = Bit("00001010")
print(bit_test)
bit_test_2 = Bit("01010101")
xor_bits = bit_test ^ bit_test_2
print(xor_bits)
xor_bits[0] = 1
print(xor_bits)
print(xor_bits ^ bit_test)