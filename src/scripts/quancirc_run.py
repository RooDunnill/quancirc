from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols.primitives import *





bit_test = Bit("00001010")
print(bit_test)
bit_test_2 = Bit("01010101")
xor_bits = bit_test ^ bit_test_2
print(xor_bits)
xor_bits[0] = 1
print(xor_bits)
print(xor_bits ^ bit_test)
test = Gate.Hadamard
qub = Qubit.q0()
qub2 = Qubit(state=[1,0])
print(qub.__dir__())
print(dir(qub))
print(qub2.__dir__())
print(dir(qub2))
test_gate = Gate(matrix=[[1,0],[0,1]])
print(n_int_length_key(10))
print(n_bit_length_key(10))
print(n_int_length_key(100))
print(n_bit_length_key(100))
print(n_hex_length_key(1000))