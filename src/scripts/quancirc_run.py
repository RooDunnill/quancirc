from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84
from ..crypto_protocols import otp
import numpy as np
import sympy as sympy
from sympy import pprint
from ..circuit.classes.symbolic_classes.symb_quant_info import *
from ..circuit.classes.symbolic_classes.symb_qubit import *

phi = sympy.symbols("phi", real=False)
a, b, c = sympy.symbols("a b c", real=False)
d, e, f = sympy.symbols("d e f", real=False)
state_1 = SymbQubit(rho=sympy.Matrix([[a, b], [sympy.conjugate(b), c]]),skip_validation=True)
state_2 = SymbQubit(rho=sympy.Matrix([[d, e], [sympy.conjugate(e), f]]),skip_validation=True)
state_1.rho = state_1.rho.subs({b:0.0, a:sympy.cos(phi)**2, c:sympy.sin(phi)**2,})
state_2.rho = state_2.rho.subs({d:sympy.sin(phi)**2, f:sympy.cos(phi)**2, e:0.0})

expression = SymbQuantInfo.trace_distance(state_1, state_2)
pprint(expression.subs({phi:0.0}))

expression = SymbQuantInfo.fidelity(state_1, state_2)
print(expression.subs({phi:0.0}))

expression = SymbQuantInfo.trace_distance_bound(state_1, state_2)
print(expression[0].subs({phi:0.0}))
print(expression[1].subs({phi:0.0}))



