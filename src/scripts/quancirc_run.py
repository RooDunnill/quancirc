from ..circuit.classes import *








test_qub = Qubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
test_qub.set_display_mode("density")
test_qub.debug()
print(test_qub)

test_qub2 = test_qub % q0
print(test_qub2)
print(test_qub2.partial_trace_gen_gen(1,0))
print(test_qub2.partial_trace_gen_gen(0,1))