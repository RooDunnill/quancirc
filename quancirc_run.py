from classes import *


attr_test_0 = q0
attr_test_0.set_display_mode("both")
attr_test_0.skip_val = True
print(attr_test_0.skip_val)
attr_test_1 = q1
attr_test_1.set_display_mode("density")
attr_sum = attr_test_0 % attr_test_1
print(attr_sum.display_mode)
print(attr_sum.skip_val)
print(type(attr_sum))
print(attr_sum)
index_test = q0 % q0 % q0
test = index_test[0]
print(test.index)
test2 = test % q0
print(test2.index)