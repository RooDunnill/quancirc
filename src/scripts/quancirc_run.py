from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84





bb84_key = bb84.gen(100)
encoded_qubits = bb84.enc(bb84_key)
measured_encryption = bb84.measure(encoded_qubits)
red_received_key, red_encoding_key = bb84.compare_basis(measured_encryption, bb84_key)
len_reduced_received_key = len(red_received_key)
half_len = int(len_reduced_received_key / 2)
received_final_key = red_received_key[half_len:]
print(received_final_key)
print(red_encoding_key[1::2])
