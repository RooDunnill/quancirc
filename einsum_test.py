import numpy as np
mat_1 = np.array([[1,0],[0,0]])

mat_2 = np.array([[0,0],[0.2,1]])

mat_3 = np.array([[0.5,0],[0,0.5]])
mat_combined = np.kron(mat_1, mat_2)
mat_combined = np.kron(mat_combined, mat_3)
print(mat_combined)
mat_reshaped = mat_combined.reshape(2,2,2,2,2,2)
print(mat_reshaped)

chars = set(['a','b','c','d','e','f'])

for a in chars:
    for b in set.difference(chars, [a]):
        print("abcdef->" + a + b + ": \n" + str(np.einsum("abcdef->" + a + b, mat_reshaped)) + "\n")