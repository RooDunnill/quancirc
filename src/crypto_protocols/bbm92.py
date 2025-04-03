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
    if n % 2 != 0:
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
            bbm92_circ.apply_gate_on_array(Z_Gate, index=index, qubit=0)
        elif bits == "10":           #psi plus
            bbm92_circ.apply_gate_on_array(Hadamard, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(CNot, index=index)
            bbm92_circ.apply_gate_on_array(X_Gate, index=index, qubit=0)
        elif bits == "11":           #psi minus
            bbm92_circ.apply_gate_on_array(Hadamard, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(CNot, index=index)
            bbm92_circ.apply_gate_on_array(X_Gate, index=index, qubit=0)
            bbm92_circ.apply_gate_on_array(Z_Gate, index=index, qubit=0)
        else:
            raise BBM92Error(f"This section of the key {bits} is not a valid input")
    print(f"Qubits now encoded into Bell states")
    encoded_qubits = bbm92_circ.download_qubit_array()
    return encoded_qubits

def measure_a(Qubit_array, verbose=True):
    if not isinstance(Qubit_array, QubitArray):
        raise BB84Error(f"Qubit array cannot be of type {type(Qubit_array)}, expected type QubitArray")
    print(f"Starting circuit a in array mode") if verbose else None
    bbm92_circ_a = Circuit(mode="array", verbose=verbose)
    print(f"Uploading received qubit array to the circuit") if verbose else None
    bbm92_circ_a.upload_qubit_array(Qubit_array)
    print(f"Generating basis measurements") if verbose else None
    basis_key_a = n_length_bit_key(len(Qubit_array))
    for i in range(len(Qubit_array)):
        if int(basis_key_a[i]) == 0:
            bbm92_circ_a.measure_states_on_array(index=i, basis="Z")
        elif int(basis_key_a[i]) == 1:
            bbm92_circ_a.measure_states_on_array(index=i, basis="X")
        else:
            raise BBM92Error(f"{basis_key_a[i]} is not a valid element of the basis key, can only be 0 or 1")
    print(f"Returning bits off of the circuit") if verbose else None
    measured_bits = bbm92_circ_a.download_bits()
    print(f"Returning qubits") if verbose else None
    measured_a_qubits = bbm92_circ_a.download_qubit_array()
    if len(basis_key_a) != len(measured_bits):
        raise BBM92Error(f"The length of the basis key must equal the length of the measured bits, not {len(basis_key_a)} and {len(measured_bits)}")
    return measured_a_qubits, ''.join(chain(*zip(basis_key_a, measured_bits.return_bits_as_str())))
    
    
def measure_b(Qubit_array, verbose=True):
    if not isinstance(Qubit_array, QubitArray):
        raise BB84Error(f"Qubit array cannot be of type {type(Qubit_array)}, expected type QubitArray")
    print(f"Starting circuit b in array mode") if verbose else None
    bbm92_circ_b = Circuit(mode="array", verbose=verbose)
    print(f"Uploading received qubit array to the circuit") if verbose else None
    bbm92_circ_b.upload_qubit_array(Qubit_array)
    print(f"Generating basis measurements") if verbose else None
    basis_key_b = n_length_bit_key(len(Qubit_array))
    for i in range(len(Qubit_array)):
        if int(basis_key_b[i]) == 0:
            bbm92_circ_b.measure_states_on_array(index=i, basis="Z")
        elif int(basis_key_b[i]) == 1:
            bbm92_circ_b.measure_states_on_array(index=i, basis="X")
        else:
            raise BBM92Error(f"{basis_key_b[i]} is not a valid element of the basis key, can only be 0 or 1")
    print(f"Returning bits off of the circuit") if verbose else None
    measured_bits = bbm92_circ_b.download_bits()
    print(f"Returning qubits") if verbose else None
    measured_b_qubits = bbm92_circ_b.download_qubit_array()
    if len(basis_key_b) != len(measured_bits):
        raise BBM92Error(f"The length of the basis key must equal the length of the measured bits, not {len(basis_key_b)} and {len(measured_bits)}")
    return measured_b_qubits, ''.join(chain(*zip(basis_key_b, measured_bits.return_bits_as_str())))


def compare_basis(key_a, key_b):
    check_bit_string(key_a)
    check_bit_string(key_b)
    key_a_len = len(key_a)
    key_b_len = len(key_b)
    if key_a_len != key_b_len:
        raise BB84Error(f"Both keys must be the same length, not of length a:{key_a} and b:{key_b}")
    key_split_len = int(key_a_len/2)
    reduced_key_a = ""
    reduced_key_b = ""
    for i in range(key_split_len):
        if key_a[2 * i] == key_b[2 * i]:
            basis = "Z" if key_a[2 * i] == "0" else "X"
            print(f"Measured in the same basis ({basis}) for qubit {i}")
            reduced_key_a += key_a[2 * i + 1]
            reduced_key_b += key_b[2 * i + 1]
    return reduced_key_a, reduced_key_b





            
