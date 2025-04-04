import numpy as np
import sympy as sympy
from sympy import pprint
from ..circuits import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84

from ..crypto_protocols import otp
from ..crypto_protocols import rsa_weak_key_gen
from ..examples import *
from ..examples.circuit_examples.generators_printer import *
from ..circuit_algorithms.grover_search_sparse import *
from ..circuits.general_circuit.utilities.fwht import *



test = Circuit(states=2, q=2)
test.apply_gate(Hadamard, qubit=0)
test.apply_gate(Hadamard, index=1, qubit=1)
qubit_array = QubitArray(q=4)
test_2 = Circuit()
print(test_2.qubit_array)
test_2.upload_qubit_array(qubit_array)
test_2.print_state()
print("\n")
print(test_2.qubit_array)
print(len(qubit_array))
test_2.measure_state(1)


test_qub = Qubit.q0(n=4)
print(test_qub[0:3])
