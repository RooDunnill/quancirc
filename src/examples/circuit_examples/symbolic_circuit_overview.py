import sympy as sp
from ...circuits.symbolic_circuit.classes import *
from ...config import *
from sympy import pprint

def symbolic_circuit_guide():
    print("=" * linewid)
    print(f"Welcome to the overview of the symbolic circuit")
    print(f"Here i will continually add new examples about what you can do and how to do it")
    print(f"The reason I implemented this class in the first place was to understand trace distance more, so here is some expressions you can do")
    phi_value = 1.0   #this value can be either a float, int or str, int and str are treated as variables while with a flaot it will actually compute the value
    phi = sp.symbols("phi", real=False)
    a, b, c = sp.symbols("a b c", real=False)
    d, e, f = sp.symbols("d e f", real=False)
    state_1 = SymbQubit(rho=sp.Matrix([[a, b], [sp.conjugate(b), c]]))
    state_2 = SymbQubit(rho=sp.Matrix([[d, e], [sp.conjugate(e), f]]))

    state_1.rho = state_1.rho.subs({b:0.0, a:sp.cos(phi)**2, c:sp.sin(phi)**2,})   #a long way of just setting them, when i could have done that when 
    state_2.rho = state_2.rho.subs({d:sp.sin(phi)**2, f:sp.cos(phi)**2, e:0.0})    #implementing class

    expression = SymbQuantInfo.trace_distance(state_1, state_2)
    pprint(f"Trace Distance: {expression.subs({phi:phi_value})}")

    expression = SymbQuantInfo.fidelity(state_1, state_2)
    print(f"Fidelity: {expression.subs({phi:phi_value})}")

    expression = SymbQuantInfo.trace_distance_bound(state_1, state_2)
    print(f"Lower bound of the trace distance: {expression[0].subs({phi:phi_value})}")
    print(f"Upper bound of the trace distance: {expression[1].subs({phi:phi_value})}")  #i believe, because these states are pure, the trace distance=upper bound
    print("\nNow lets try some matrix multiplication:")
    print(f"|0> matrix multipled by a phase gate of phase theta: {P_Gate_symb @ q0_symb}")
    print(f"|i> matrix multipled by a T Gate: {T_Gate_symb @ qpi_symb}")          #I is sympys way of representing the imaginary number i

    print(f"Lets make a mixed state:")
    mixed_state = SymbQubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
    print(f"The mixed state: {mixed_state}")
    print(f"The mixed state multiplied by an X rotation gate of angle theta: {Rotation_x_symb @ mixed_state}")   #bit of a mess, doesnt actually do anything
    print(f"Here is a general Qubit: {qgen_symb}")
    print(f"You can set the values as such: {qgen_symb.subs({sp.symbols("alpha"):1})}")
    print(f"Lets tensor product two general qubits together to see what that looks like:")
    print(f"{SymbQubit.gen() % SymbQubit.gen()}")


if __name__ == "__main__":
    symbolic_circuit_guide()