from qc_errors import MixinError
class LinearMixin:
    array_name = None

    def _combine_attributes(self, other, op = "+"):
        kwargs = {}
        if hasattr(self, "name") and hasattr(other, "name"):
            kwargs["name"] = f"{self.name} {op} {other.name}"

        if hasattr(self, "info") and hasattr(other, "info"):
            kwargs["info"] = f"Combined gate of {self.info} and {other.info}"
        return kwargs

    def __add__(self, other):
        if isinstance(other, self.__class__) and self.array_name:   #only works if theyre the same class and they exist
            new_array = getattr(self, self.array_name) + getattr(other, self.array_name)
            kwargs = {self.array_name: new_array}
            kwargs.update(self._combine_attributes(other, op = "+"))
            return self.__class__(**kwargs)
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __iadd__(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            setattr(self, self.array_name, getattr(self, self.array_name) + getattr(other, self.array_name))   #adds to that attribute, ie self.matrix for gate
            return self
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __sub__(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            new_array = getattr(self, self.array_name) - getattr(other, self.array_name)
            kwargs = {self.array_name: new_array}
            kwargs.update(self._combine_attributes(other, op = "-"))
            return self.__class__(**kwargs)
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")
    
    def __isub__(self, other):
        if isinstance(other, self.__class__) and self.array_name:
            setattr(self, self.array_name, getattr(self, self.array_name) - getattr(other, self.array_name))
            return self
        raise MixinError(f"The classes do not match or the array is not defined. They are of types {type(self.__class__)} and {type(other.__class__)}")