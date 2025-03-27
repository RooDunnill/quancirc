from ..circuit.classes import *
from ..circuit.classes.lightweight_circuit import *
from ..circuit_algorithms.grover_search import *
from ..crypto_protocols import *
from ..crypto_protocols import bb84
from ..crypto_protocols import otp
import numpy as np




trace_state_1 = Qubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
QuantInfo.bloch_plotter(trace_state_1)


state_1 = Qubit(state=[1,0])
QuantInfo.bloch_plotter(Gate.Rotation_Y(np.pi/4) @ state_1)
