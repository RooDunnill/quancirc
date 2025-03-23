from ..circuit.classes import *

qub_0 = q1 % qp
qub_p = q1 % qp
qub_p.set_display_mode("both")
print(qub_p)
print(QuantInfo.trace_distance_bound(qub_0, qub_p))
print(QuantInfo.trace_distance(qub_0, qub_p))
QuantInfo.two_state_info(qub_0, qub_p)