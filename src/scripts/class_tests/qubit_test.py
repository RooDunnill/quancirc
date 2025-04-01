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

