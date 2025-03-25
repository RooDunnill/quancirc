from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *

grover_search(16, n=16, verbose=False)
test = grover_search(16, verbose=False)
print(test)

