import numpy as np
import sympy as sympy
from sympy import pprint
from ...circuits import *
import random

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
sparse_qub = q0_symb % q0_symb % q0_symb % q0_symb % qm_symb
test = sparse_qub % sparse_qub
print(type(test.rho))
sparse_qub = q0_lw % q0_lw % q0_lw % q0_lw % qm_lw
test = sparse_qub % sparse_qub
print(type(test.state))
sparse_qub = q0 % q0 % q0 % q0 % q0 % q0
test = sparse_qub % sparse_qub
print(type(test.rho))

test_1 = q0 @ q0
print(test_1)
print(type(test_1.rho))
large_qub = Qubit.q0(n=10)
test_2 = large_qub @ large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=9)
test_2 = large_qub @ large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=8)
test_2 = large_qub @ large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=7)
test_2 = large_qub @ large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=3)
test_2 = large_qub % large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=4)
test_2 = large_qub % large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=5)
test_2 = large_qub % large_qub
print(test_2)
print(type(test_2.rho))
large_qub = Qubit.q0(n=6)
test_2 = large_qub % large_qub
print(test_2)
print(type(test_2.rho))
random_qubit_list = [q0, q1, qp, qm, qpi, qmi]
test_qub = q0 % qp
for _ in range(10):
    rand_qub = random.choice(random_qubit_list)
    test_qub %= rand_qub
    print(f"modulus")
    print(type(test_qub.rho))
    test_qub @= test_qub
    print(f"mat mul")
    print(type(test_qub.rho))

test_qub = q0 % q0
for _ in range(10):
    rand_qub = q0
    test_qub %= rand_qub
    print(f"modulus")
    print(test_qub)
    print(type(test_qub.rho))
    test_qub @= test_qub
    print(f"mat mul")
    print(test_qub)
    print(type(test_qub.rho))
    if test_qub.rho[0,1] == 0:
        print(f"success")
