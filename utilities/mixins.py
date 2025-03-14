from .qc_errors import MixinError
import numpy as np

array_name = None

__all__ = ["DirectSumMixin", "LinearMixin", "BaseMixin"]

def combine_attributes(self, other, op = "+"):
        """Allows the returned objects to still return name and info too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):   #takes the name of the two objects and combines them accordingly
            kwargs["name"] = f"{self.name} {op} {other.name}"

        if hasattr(self, "info") and hasattr(other, "info"):   #takes the info of the two objects and combines them
            kwargs["info"] = f"Combined gate of {self.info} and {other.info}"
        return kwargs



class BaseMixin:
    def __eq__(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            return self.array_name == other.array_name
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __ne__(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            return self.array_name != other.array_name
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")

class DirectSumMixin(BaseMixin):
    """Allows for the & dunder method to be updated for the direct sum, currently only allowed for the gate class"""
    def direct_sum(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            new_dim: int = self.dim + other.dim
            new_length: int = new_dim**2
            new_mat = np.zeros(new_length,dtype=np.complex128)
            
            for i in range(self.dim):
                for j in range(self.dim):                   #a lot more elegant
                    new_mat[j+new_dim*i] += self.matrix[j+self.dim*i]
            for i in range(other.dim):     #although would be faster if i made a function to apply straight
                for j in range(other.dim):    #to individual qubits instead
                    new_mat[self.dim+j+self.dim*new_dim+new_dim*i] += other.matrix[j+other.dim*i]
            return new_mat 
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
        
    def __and__(self, other):    #this is the direct sum, mostly used for creating CNOT gates
        if isinstance(other, self.__class__) and self.array_name:
            new_mat = self.direct_sum(other)
            kwargs = {self.array_name: new_mat}
            kwargs.update(combine_attributes(self, other, op = "&"))
            return self.__class__(**kwargs)
        
    
    def __iand__(self, other):
            new_mat = self.direct_sum(other)
            setattr(self, self.array_name, new_mat)
            kwargs = {}
            kwargs.update(combine_attributes(self, other, op="&"))
            vars(self).update(kwargs)
            return self



class LinearMixin(BaseMixin):
    """Used to store some dunder methods that are used a lot in mostly 
        class Gate, Density and Qubit to allow for easy reading and for
        the main file not to get too large.
        The file takes the predefined array_name in each class, rho for density
        vector for Qubit and matrix for Gate and then puts it through the called function
        to output an attribute back to that.
        The combine attributes allows for the newly created object to also have a name and info."""
    

    def __add__(self, other):
        """adds together the array of each object, ie matrices for gates and vectors for qubits"""
        if isinstance(other, self.__class__) and self.array_name:   #only works if theyre the same class and they exist
            new_array = getattr(self, self.array_name) + getattr(other, self.array_name)
            kwargs = {self.array_name: new_array}
            kwargs.update(combine_attributes(self, other, op = "+"))
            return self.__class__(**kwargs)
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __iadd__(self, other):
        """iterative addition"""
        if isinstance(other, self.__class__) and self.array_name:
            setattr(self, self.array_name, getattr(self, self.array_name) + getattr(other, self.array_name))   #adds to that attribute, ie self.matrix for gate
            return self
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __sub__(self, other):
        """subtracts the arrays of each object"""
        if isinstance(other, self.__class__) and self.array_name:
            new_array = getattr(self, self.array_name) - getattr(other, self.array_name)
            kwargs = {self.array_name: new_array}
            kwargs.update(combine_attributes(self, other, op = "-"))
            return self.__class__(**kwargs)
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __isub__(self, other):
        """iterative subtraction"""
        if isinstance(other, self.__class__) and self.array_name:
            setattr(self, self.array_name, getattr(self, self.array_name) - getattr(other, self.array_name))
            return self
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")