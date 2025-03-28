from ...crypto_protocols import *
from ...crypto_protocols import bb84
import numpy as np


def bb84_example():
    print(f"An example of a noiseless BB84 protocol in the absence of Eve")
    bb84_key = bb84.gen_key(10)
    encoded_qubits = bb84.enc(bb84_key)
    measured_encryption = bb84.measure(encoded_qubits)
    red_received_key, red_encoding_key = bb84.compare_basis(measured_encryption, bb84_key)

    print(f"(Alice) BB84 Original Key: {bb84_key}")
    basis_prep, state = bb84.split_odd_even(bb84_key)
    print(f"(Alice) Start basis prep: {basis_prep}")
    print(f"(Alice) Start states: {state}")
    print(f"(Alice) Reduced original BB84 Key: {red_encoding_key}")
    send_basis, send_key = bb84.split_odd_even(red_encoding_key)
    receive_basis, receive_key = bb84.split_odd_even(red_received_key)
    print(f"(Bob) The final measured bases: {receive_basis}")
    print(f"(Bob) The final received key: {receive_key}")
    print(f"Complete for no interferrence if:\n(Alice) key:{send_key}= (Bob) key:{receive_key}  {np.all(send_key==receive_key)}")
    print(f"(Alice) basis:{send_basis}= (Bob) basis:{receive_basis} {np.all(send_basis==receive_basis)}")
    print(f"")

if __name__ == "__main__":
    bb84_example()