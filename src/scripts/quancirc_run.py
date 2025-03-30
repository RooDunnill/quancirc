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


print(su_N_generators(4))
print(su_N_generators(8))
print(su_N_generators(128))