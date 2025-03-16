import numpy as np
from .mixins import LinearMixin, combine_attributes
from .qc_errors import TensorError, MatMulError

array_name = None

class LinAlgMixin(LinearMixin):

    def tensor_product(self, other):
        if isinstance(other, np.ndarray):                 #used for when you just need to compute it for a given array, this is for creating seperable states
            other_dim = len(other)
            new_length: int = self.dim*other_dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other_dim):          #iterates up and down the second ket
                    new_vector[j+(i * other_dim)] += self.vector[i]*other[j] #adds the values into
            self.dim = new_length
            self.vector = new_vector
            return self.vector    #returns a new Object with a new name too
        elif self.array_name == "vector" and self.array_name:
            new_length: int = self.dim*other.dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other.dim):          #iterates up and down the second ket
                    new_vector[j+(i * other.dim)] += self.vector[i]*other.vector[j] #adds the values into each element of the vector
            return new_vector
        elif isinstance(other, self.__class__) and self.array_name:
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
        new_array = self.tensor_product(other)
        kwargs = {self.array_name: new_array}
        kwargs.update(combine_attributes(self, other, op = "@"))
        if isinstance(other, self.__class__) and self.array_name:
            return self.__class__(**kwargs)
        elif isinstance(other, np.ndarray) and self.array_name:
            return self.vector
      