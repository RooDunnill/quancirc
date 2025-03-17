import sys; sys.path.append("..")
from classes import *
from utilities import *
test = Qubit(state=[0,1])
test.set_display_mode("both")
print(test)
test2 = Qubit(rho=[[1,0],[0,0]])
test2.set_display_mode("both")
print(test2)
print(q0)
test3 = q0 @ q0
test3.set_display_mode("both")
print(test3)
print(q0 & q1)
print(q1 - q0)
print(q1 + q0)