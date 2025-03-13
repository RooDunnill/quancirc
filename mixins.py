from qc_errors import MixinError
class LinearMixin:
    """Used to store some dunder methods that are used a lot in mostly 
        class Gate, Density and Qubit to allow for easy reading and for
        the main file not to get too large.
        The file takes the predefined array_name in each class, rho for density
        vector for Qubit and matrix for Gate and then puts it through the called function
        to output an attribute back to that.
        The combine attributes allows for the newly created object to also have a name and info."""
    array_name = None

    def _combine_attributes(self, other, op = "+"):
        """Allows the returned objects to still return name and info too"""
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):
            kwargs["name"] = f"{self.name} {op} {other.name}"

        if hasattr(self, "info") and hasattr(other, "info"):
            kwargs["info"] = f"Combined gate of {self.info} and {other.info}"
        return kwargs

    def __add__(self, other):
        """adds together the array of each object, ie matrices for gates and vectors for qubits"""
        if isinstance(other, self.__class__) and self.array_name:   #only works if theyre the same class and they exist
            new_array = getattr(self, self.array_name) + getattr(other, self.array_name)
            kwargs = {self.array_name: new_array}
            kwargs.update(self._combine_attributes(other, op = "+"))
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
            kwargs.update(self._combine_attributes(other, op = "-"))
            return self.__class__(**kwargs)
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __isub__(self, other):
        """iterative subtraction"""
        if isinstance(other, self.__class__) and self.array_name:
            setattr(self, self.array_name, getattr(self, self.array_name) - getattr(other, self.array_name))
            return self
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")