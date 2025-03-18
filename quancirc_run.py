import sys; sys.path.append("..")
from classes import *
from utilities import *
state_1 = q0 @ q0
state_2 = q1 @ q1

QuantInfo.two_state_info(state_1, state_2)
