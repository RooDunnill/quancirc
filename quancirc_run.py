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
measure_test_qubit = q1 @ q0
measure_test = Measure(state=measure_test_qubit).list_probs()
print(measure_test)
pm_state = Measure(state=measure_test_qubit).measure_state()
print(type(pm_state))
pm_state.set_display_mode("both")
print(pm_state)
print(pm_state.display_mode)
measure_test_qub1 = Measure(state = measure_test_qubit[1]).list_probs()
print(measure_test_qub1)
pm_state_qub1 = Measure(state = measure_test_qubit[1]).measure_state()
pm_state_qub1.set_display_mode("both")
print(pm_state_qub1)
measure_test_qub0 = Measure(state = measure_test_qubit[0]).list_probs()
print(measure_test_qub0)
pm_state_qub0 = Measure(state = measure_test_qubit[0]).measure_state()
pm_state_qub0.set_display_mode("both")
print(pm_state_qub0)