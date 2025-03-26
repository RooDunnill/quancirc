import numpy as np
import scipy as sp
from ..circuit.classes.circuit import *
from .primitives import *
from ..circuit.classes.qubit_array import *
from .crypto_utilities import *
from ..circuit.classes.gate import *



def split_into_2_bits(bit_string):
    return [bit_string[i:i+2] for i in range(0, len(bit_string), 2)]

def gen(n, verbose=True):
    print(f"Generating a key length of {2*n} bits") if verbose else None
    return n_length_bit_key(2*n)

def enc(key: str, verbose=True):
    if is_bit_string(key) and len(key) % 2 == 0:
        print(f"Creating string of qubits to send") if verbose else None
        qubit_string = QubitArray(qubit_num=len(key) // 2)
        print(f"Starting the circuit in array mode") if verbose else None
        bb84_circ = Circuit(mode="array", verbose=verbose)
        print(f"Uploading qubit array to the circuit") if verbose else None
        bb84_circ.upload_qubit_array(qubit_string)
        key_list = split_into_2_bits(key)
        print(f"Now encoding the qubits into their constituent states") if verbose else None
        for index, bits in enumerate(key_list):
            if bits == "00":
                pass
            elif bits == "01":
                bb84_circ.apply_gate_on_array(X_Gate, index)
            elif bits == "10":
                bb84_circ.apply_gate_on_array(Hadamard, index)
            elif bits == "11":
                bb84_circ.apply_gate_on_array(X_Gate, index)
                bb84_circ.apply_gate_on_array(Hadamard, index)
            else:
                raise BB84Error(f"This section of the key {bits} is not a valid input")
        print(f"Qubits now encoded") if verbose else None
        encoded_qubits = bb84_circ.download_qubit_array()
        return encoded_qubits
    
def measure(Qubit_array, verbose=True):
    bb84_circ_rec = Circuit(mode="array", verbose=verbose)
    bb84_circ_rec.upload_qubit_array(Qubit_array)
    basis_key = n_length_bit_key(len(Qubit_array))
    key_provisional = ""
    for i in range(len(Qubit_array)):
        if int(basis_key[i]) == 0:
            bb84_circ_rec.measure_states_on_array(index=i, basis="X")
            key_provisional += "0"
        elif int(basis_key[i]) == 1:
            bb84_circ_rec.measure_states_on_array(index=i, basis="Z")
            key_provisional += "1"
        else:
            raise BB84Error(f"{basis_key[i]} is not a valid element of the basis key, can only be 0 or 1")
    print(key_provisional)
    measured_bits = bb84_circ_rec.return_bits
    print(bb84_circ_rec.bits)
    print(measured_bits)
    print(type(measured_bits))
    print(measured_bits.bit_string)
    return basis_key + key_provisional

def compare_basis(measure_key, encoding_key):
    measure_key_len = len(measure_key)
    encoding_key_len = len(encoding_key)
    if measure_key_len != encoding_key_len:
        raise BB84Error(f"Both keys must be the same length, not of length measure:{measure_key_len} and encoding:{encoding_key_len}")
    key_split_len = int(measure_key_len/2)
    measure_basis = measure_key[:key_split_len]
    outcome_key = measure_key[key_split_len:]
    reduced_measure_basis = ""
    reduced_outcome_key = ""
    reduced_encoding_key = ""
    for i in range(key_split_len):
        if measure_basis[i] != encoding_key[2* i]:
            reduced_measure_basis += measure_basis[i]
            reduced_outcome_key += outcome_key[i]
            reduced_encoding_key += encoding_key[i:i+1]
    return reduced_measure_basis + reduced_outcome_key, reduced_encoding_key





            
