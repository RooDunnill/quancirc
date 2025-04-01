import numpy as np
import sympy as sympy
from sympy import pprint
from ...circuits import *

print(q0 + q0)
print(q0 - q0)
print(q0 % q0)
print(q0 @ q0)
print(q0 / 2)
print(q0 * 2)
print(qm + qm)
print(qm - qm)
print(qm % qm)
print(qm @ qm)
print(qm / 2)
print(qm * 2)
qub_3 = qm % q1 % qpi
qub_3_2 = qmi % q0 % qpi
print(qub_3_2 % qub_3)
print(qub_3_2 @ qub_3)
sparse_qub = q0 % q0 % q0 % q0
print(sparse_qub % sparse_qub % sparse_qub)


print(q0_lw + q0_lw)
print(q0_lw - q0_lw)
print(q0_lw % q0_lw)
print(q0_lw / 2)
print(q0_lw * 2)
print(qm_lw + qm_lw)
print(qm_lw - qm_lw)
print(qm_lw % qm_lw)
print(qm_lw / 2)
print(qm_lw * 2)
qub_3 = qm_lw % q1_lw % qpi_lw
qub_3_2 = qmi_lw % q0_lw % qpi_lw
print(qub_3_2 % qub_3)
print(qub_3_2 + qub_3)
sparse_qub = q0_lw % q0_lw % q0_lw % q0_lw


print(q0_symb + q0_symb)
print(q0_symb - q0_symb)
print(q0_symb % q0_symb)
print(q0_symb @ q0_symb)
print(q0_symb / 2)
print(q0_symb * 2)
print(qm_symb + qm_symb)
print(qm_symb - qm_symb)
print(qm_symb % qm_symb)
print(qm_symb @ qm_symb)
print(qm_symb / 2)
print(qm_symb * 2)
print(qgen_symb + qgen_symb)
print(qgen_symb - qgen_symb)
print(qgen_symb % qgen_symb)
print(qgen_symb @ qgen_symb)
print(qgen_symb / 2)
print(qgen_symb * 2)
qub_3 = qm_symb % q1_symb % qpi_symb
qub_3_2 = qmi_symb % q0_symb % qpi_symb
print(qub_3_2 % qub_3)
print(qub_3_2 @ qub_3)
sparse_qub = q0_symb % q0_symb % q0_symb % q0_symb
