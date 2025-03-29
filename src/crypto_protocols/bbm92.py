import numpy as np
import scipy as sp
from ..circuits.general_circuit import *
from .primitives import *
from .crypto_utilities import *
from itertools import chain

__all__ = ["gen_key", "enc", "measure", "compare_basis" ]

def split_odd_even(bit_string):
    check_bit_string(bit_string)
    return bit_string[0::2], bit_string[1::2]

def split_2b_chunks(bit_string):
    check_bit_string(bit_string)
    return [bit_string[i:i+2] for i in range(0, len(bit_string), 2)]

def split_bit_string(bit_string):
    check_bit_string(bit_string)
    length = len(bit_string)
    if length % 2 != 0:
        raise PrimitiveError(f"The inputted bit string must be even")
    half_len = int(length / 2)
    return bit_string[:half_len], bit_string[half_len:]


def gen_key(n, verbose=True):
    if not isinstance(n, int):
        raise BBM92Error(f"n cannot be of type {type(n)}, expected type int")
    if n < 0:
        raise BBM92Error(f"n ({n}), cannot be less than or equal to 0, must be positive integer")
    if len(n) % 2 != 0:
        raise BBM92Error(f"The inputted key must have an even amount of bits")
    print(f"Generating a key length of {2*n} bits") if verbose else None
    return n_length_bit_key(2*n)

def gen_bell_states(key, verbose=True):
    check_bit_string(key)
    print(f"Creating string of qubits to create into Bell states") if verbose else None
    bell_string = QubitArray(q=len(key), qubit_size=2)
    print(f"Starting the circuit in array mode") if verbose else None
    bbm92_circ = Circuit(mode="array", verbose=verbose)
    print(f"Uploading qubit array to the circuit") if verbose else None
    bbm92_circ.upload_qubit_array(bell_string)
    key_list = split_2b_chunks(key)
    for index, bits in enumerate(key_list):
        if bits == "00":             #phi plus
            bbm92_circ.apply_gate_on_array(Hadamard, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(CNot, index=index)
        elif bits == "01":           #phi minus
            bbm92_circ.apply_gate_on_array(Hadamard, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(CNot, index=index)
            bbm92_circ.apply_gate_on_array(Z_Gate, index=index)
        elif bits == "10":           #psi plus
            bbm92_circ.apply_gate_on_array(Hadamard, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(CNot, index=index)
            bbm92_circ.apply_gate_on_array(X_Gate, index=index)
        elif bits == "11":
            bbm92_circ.apply_gate_on_array(Hadamard, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(CNot, index=index)
            bbm92_circ.apply_gate_on_array(X_Gate, index=index)
            bbm92_circ.apply_gate_on_array(Z_Gate, index=index)
        else:
            raise BBM92Error(f"This section of the key {bits} is not a valid input")
    print(f"Qubits now encoded into Bell states")
    encoded_qubits = bbm92_circ.download_qubit_array()
    return encoded_qubits

def measure_a(Qubit_array, verbose=True):
    if not isinstance(Qubit_array, QubitArray):
        raise BB84Error(f"Qubit array cannot be of type {type(Qubit_array)}, expected type QubitArray")
    print(f"Starting the circuit in array mode") if verbose else None
    bbm92_circ_a = Circuit(mode="array", verbose=verbose)
    print(f"Uploading received qubit array to the circuit") if verbose else None
    bbm92_circ_a.upload_qubit_array(Qubit_array)
    print(f"Generating basis measurements") if verbose else None
    basis_key_a = n_length_bit_key(len(Qubit_array))
    for i in range(len(Qubit_array)):
        if int(basis_key_a[i]) == 0:
            bbm92_circ_a.measure_states_on_array(index=i, qubit=0, basis="Z")
        elif int(basis_key_a[i]) == 1:
            bbm92_circ_a.measure_states_on_array(index=i, qubit=0, basis="X")
        else:
            raise BBM92Error(f"{basis_key_a[i]} is not a valid element of the basis key, can only be 0 or 1")
    print(f"Returning bits off of the circuit") if verbose else None
    measured_bits = bbm92_circ_a.return_bits()
    print(f"Returning qubits") if verbose else None
    measured_a_qubits = bbm92_circ_a.download_qubit_array()
    return measured_a_qubits, ''.join(chain(*zip(basis_key_a, measured_bits.return_bits_as_str())))
    
def measure_b(Qubit_array, verbose=True):
    if not isinstance(Qubit_array, QubitArray):
        raise BB84Error(f"Qubit array cannot be of type {type(Qubit_array)}, expected type QubitArray")
    print(f"Starting the circuit in array mode") if verbose else None
    bbm92_circ_b = Circuit(mode="array", verbose=verbose)
    print(f"Uploading received qubit array to the circuit") if verbose else None
    bbm92_circ_b.upload_qubit_array(Qubit_array)
    print(f"Generating basis measurements") if verbose else None
    basis_key_b = n_length_bit_key(len(Qubit_array))
    for i in range(len(Qubit_array)):
        if int(basis_key_b[i]) == 0:
            bbm92_circ_b.measure_states_on_array(index=i, qubit=1, basis="Z")
        elif int(basis_key_b[i]) == 1:
            bbm92_circ_b.measure_states_on_array(index=i, qubit=1, basis="X")
        else:
            raise BBM92Error(f"{basis_key_b[i]} is not a valid element of the basis key, can only be 0 or 1")
    print(f"Returning bits off of the circuit") if verbose else None
    measured_bits = bbm92_circ_b.return_bits()
    print(f"Returning qubits") if verbose else None
    measured_a_qubits = bbm92_circ_b.download_qubit_array()
    return measured_a_qubits, ''.join(chain(*zip(basis_key_b, measured_bits.return_bits_as_str())))





def measure(Qubit_array, verbose=True):
    if not isinstance(Qubit_array, QubitArray):
        raise BB84Error(f"Qubit array cannot be of type {type(Qubit_array)}, expected type QubitArray")
    print(f"Starting the circuit in array mode") if verbose else None
    bb84_circ_rec = Circuit(mode="array", verbose=verbose)
    print(f"Uploading received qubit array to the circuit") if verbose else None
    bb84_circ_rec.upload_qubit_array(Qubit_array)
    print(f"Generating basis measurements") if verbose else None
    basis_key = n_length_bit_key(len(Qubit_array))
    for i in range(len(Qubit_array)):
        if int(basis_key[i]) == 0:
            bb84_circ_rec.measure_states_on_array(index=i, basis="Z")
        elif int(basis_key[i]) == 1:
            bb84_circ_rec.measure_states_on_array(index=i, basis="X")
        else:
            raise BB84Error(f"{basis_key[i]} is not a valid element of the basis key, can only be 0 or 1")
    print(f"Returning bits off of the circuit") if verbose else None
    measured_bits = bb84_circ_rec.return_bits()
    return ''.join(chain(*zip(basis_key, measured_bits.return_bits_as_str())))

def compare_basis(received_key, encoding_key):
    check_bit_string(received_key)
    check_bit_string(encoding_key)
    received_key_len = len(received_key)
    encoding_key_len = len(encoding_key)
    if received_key_len != encoding_key_len:
        raise BB84Error(f"Both keys must be the same length, not of length measure:{received_key_len} and encoding:{encoding_key_len}")
    key_split_len = int(received_key_len/2)
    reduced_encoding_key = ""
    reduced_received_key = ""
    for i in range(key_split_len):
        if received_key[2 * i] == encoding_key[2 * i]:
            print(f"Measured in the same basis for qubit {i}")
            reduced_received_key += received_key[2 * i:2 * i + 2]
            reduced_encoding_key += encoding_key[2 * i:2 * i + 2]
    return reduced_received_key, reduced_encoding_key





            
