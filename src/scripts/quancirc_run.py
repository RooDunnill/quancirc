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


super_sparse_qub = Qubit.q0(n=2) % qp % qp % qp % qp % qp
test = super_sparse_qub % super_sparse_qub
print(type(test.rho))
print(count_zeros(test.rho))
print(test.rho.count_nonzero())
print(test.rho)
