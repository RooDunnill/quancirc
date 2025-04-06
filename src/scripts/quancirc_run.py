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




test_qubit = q0 % q1
test_qubit = test_qubit % qm
test_qubit = test_qubit * 2
test_qubit[0] * 2
test_qubit.print_history()
