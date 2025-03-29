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



test_mixed_state = Qubit.create_mixed_state([q0,q1],[0.5,0.5])
print(test_mixed_state)

test = SymbQubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
print(test)

test2 = SymbQubit(state=[[1,0],[0,1],[sp.sqrt(0.5),sp.sqrt(0.5)]], weights=[0.2,0.2,0.6])
print(test2)

test3 = SymbQubit(state=[[1,0],[0,1],[sp.sqrt(0.5),sp.sqrt(0.5)]], weights=[sp.symbols("alpha"),0.2,0.6])
print(test3)