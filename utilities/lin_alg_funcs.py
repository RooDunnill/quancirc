import numpy as np
from .mixins import LinearMixin, combine_attributes
from .qc_errors import TensorError, MatMulError

array_name = None

class LinAlgMixin(LinearMixin):

    def tensor_product(self, other):
        new_length: int = self.length*other.length
        new_array = np.zeros(new_length,dtype=np.complex128)
        new_dim: int = self.dim * other.dim
        self_array = getattr(self, self.array_name)
        other_array = getattr(other, other.array_name)
        for m in range(self.dim):
            for i in range(self.dim):
                for j in range(other.dim):             #4 is 100 2 is 10
                    for k in range(other.dim):   #honestly, this works but is trash and looks bad
                        index = k+j*new_dim+other.dim*i+other.dim*new_dim*m
                        new_array[index] += self_array[i+self.dim*m]*other_array[k+other.dim*j]
        return new_array

    def __matmul__(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            new_array = self.tensor_product(other)
            kwargs = {self.array_name: new_array}
            kwargs.update(combine_attributes(self, other, op = "@"))
            return self.__class__(**kwargs)
      