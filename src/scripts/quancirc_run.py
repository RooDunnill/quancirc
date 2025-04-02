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


noise_circuit = Circuit(q=3)

noise_circuit.apply_local_channel_to_state(channel="X", prob=0.5)

noise_circuit.print_state()