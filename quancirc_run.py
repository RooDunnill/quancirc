from classes import *


test_state = q0 @ q0 @ q0
test_state[0] = qp
test_state[1] = qp
test_state[0] = Measure(test_state[0]).measure_state()
test_state[1] = Measure(test_state[1]).measure_state()
print(test_state)
print(range(0,2))
print(range(2))
print(range(0,10))
range(2)