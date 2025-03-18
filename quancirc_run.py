from classes import *
import utilities.config
state_1 = q0 @ q0
state_2 = q1 @ q1

QuantInfo.two_state_info(state_1, state_2)
print(Hadamard)
print(Hadamard @ X_Gate)
print(q0)
print(q0 @ q0)
q0.set_display_mode("both")
print(q0)

and_test1 = Bit(4, val=0)
and_test2 = Bit(4, val=1)
and_test2[1] = 0
and_test1[2] = 1
print(and_test1 & and_test2)
print(and_test1 | and_test2)
print(and_test1 ^ and_test2)