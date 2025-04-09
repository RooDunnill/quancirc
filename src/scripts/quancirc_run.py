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
from ..circuit_algorithms.grover_search_sparse import *
from ..circuits.general_circuit.utilities.fwht import *




test_qutrit = Qutrit(state=[1,0,0])
print(test_qutrit)
print(qt1 % qt2)
test_qutrit_2 = qt0 % qt1 % qt2
print(test_qutrit_2[1])
print(test_qutrit_2[0:2])
test_qutrit_2.print_history()
print(test_qutrit_2 * 10)
test_qutrit_2.print_history()
test_qubit_2 = q0 % q1 % qp
test_qubit_2.print_history()
print(test_qubit_2 * 10)
test_qubit_2.print_history()
print(test_qutrit_2 @ test_qutrit_2)
test_qubit_2.set_display_mode("ind_qub")
print(test_qubit_2)
test_gate = QutritGate.gm1(np.pi/2)
print(test_gate @ test_qutrit)