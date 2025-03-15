from .qc_errors import MixinError
import numpy as np
from rich.console import Console
from rich.theme import Theme
from .layout_funcs import format_ket_notation



custom_theme = Theme({"qubit":"#587C53",                 #Fern Green
                      "prob_dist":"#3C4E35",             #Dark Forest Green
                      "gate":"#3E5C41",                  #Forest Green
                      "density":"#4D5B44",               #Olive Green
                      "info":"#7E5A3C",                  #Earthy Brown
                      "error":"dark_orange",
                      "measure":"#3B4C3A",               #Deep Moss Green
                      "grover_header":"#7D9A69",         #Sage Green
                      "circuit_header":"#465C48",        #Muted Green
                      "main_header":"#4B7A4D"})          #Vibrant moss Green

console = Console(style="none",theme=custom_theme, highlight=False)


array_name = None

__all__ = ["DirectSumMixin", "LinearMixin", "BaseMixin", "StrMixin", "custom_theme", "console"]

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
    
class StrMixin(BaseMixin):
    prec: int = 3

    def __str__(self) -> str:
        return "\n".join(self.format_str())

    def __repr__(self) -> str:
        return self.__str__()
    
    def __rich__(self) -> str:
        return self.__str__()
    
    def format_str(self) -> list[str]:
        np.set_printoptions(precision=self.prec, suppress=True, floatmode="fixed")
      
        if self.array_name == "probs":
            return self.format_measure()
        elif self.array_name == "matrix":
            return self.format_gate()
        elif self.array_name == "rho":
            return self.format_density()
        elif self.array_name == "kets":
            return self.format_grover()
        elif self.array_name == "vector":
            return self.format_qubit()
        elif isinstance(self, np.ndarray):
            return self.format_ndarray()
        return self.format_default()
    
    def format_measure(self):
        if self.measure_type == "projective":
            ket_mat = range(self.dim)
            num_bits = int(np.ceil(np.log2(self.dim)))
            np.set_printoptions(linewidth=(10))
            output = [f"[measure]{self.name}[/measure]"]
            for ket_val, prob_val in zip(ket_mat, self.list_probs()):
                output.append(f"[measure]|{bin(ket_val)[2:].zfill(num_bits)}>  {100*prob_val:.{self.prec}f}%[/measure]")
            return output
        return [f"[measure]{self.name}[/measure]"]
    
    def format_gate(self) -> list[str]:
        if self.dim < 5:
            np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * self.dim)
        elif self.dim < 9:
            np.set_printoptions(linewidth=3 + (8 + 2 * (self.prec - 1)) * self.dim, precision=self.prec - 1)
        else:
            np.set_printoptions(linewidth=3 + (8 + 2 * (self.prec - 2)) * self.dim, precision=self.prec - 2)
        return [f"[gate]{self.name}\n{self.matrix}[/gate]"]
    
    def format_density(self) -> list[str]:
        if self.dim < 5:
            np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * self.dim)
        elif self.dim < 9:
            np.set_printoptions(linewidth=3 + (8 + 2 * (self.prec - 1)) * self.dim, precision=self.prec - 1)
        else:
            np.set_printoptions(linewidth=3 + (8 + 2 * (self.prec - 2)) * self.dim, precision=self.prec - 2)
        return [f"[density]{self.name}\n{self.rho}[/density]"]
    
    def format_qubit(self) -> list[str]:
        np.set_printoptions(linewidth=10)
        if self.state_type == "mixed":
            if isinstance(self.vector[0], np.ndarray):
                return f"[qubit]{self.name}[/qubit]\n[qubit]Vectors:\n {self.vector}[/qubit]\n and Weights:\n {self.weights}"
            else:
                print_out = f"{self.name}\nWeights: {self.weights}\n"
                for i in self.vector:
                    print_out += f"State: {i}\n"
                return print_out
        return [f"[qubit]{self.name}\n{self.vector}[/qubit]"]


    def format_np_array(self):
        length = len(self)
        if length < 17:
            np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * np.sqrt(length))
        elif length < 65:
            self.prec = self.prec - 1
            np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * np.sqrt(length), precision=self.prec)
        else:
            self.prec = self.prec - 2
            np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * np.sqrt(length), precision=self.prec)
        return [f"[gate]{self}\n[\gate]"]
    
    def format_grover(self) -> list[str]:
        print_out_kets = format_ket_notation(self.results, type="topn", num_bits=int(np.ceil(self.n)), precision = (3 if self.n < 20 else 6))
        return [f"[prob_dist]{self.name}\n{print_out_kets}[/prob_dist]"]

    def format_default(self):
        return [f"test"]

