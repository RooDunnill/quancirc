import numpy as np
import sympy as sympy
from sympy import pprint
from ..circuit import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84
from ..crypto_protocols import otp



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

print(q0_symb)
test_state = SymbQubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
print(test_state.state)
print(test_state)
print(test_state.state[2])
print(type(test_state.state[0]))
print(type(test_state.state))
print(Hadamard_symb @ test_state)
print(Hadamard_symb @ q0_symb)
print(P_Gate_symb @ qp_symb)