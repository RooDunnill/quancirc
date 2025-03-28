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

print(rsa_key_gen(83719, 174121, 65537))