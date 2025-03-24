import scipy as sp
from scipy import sparse
import numpy as np
Id_lower = sparse.eye_array(8)
Id_upper = sparse.eye_array(8)
print(Id_lower)
print(Id_upper)
print(sparse.kron(Id_lower, Id_upper))