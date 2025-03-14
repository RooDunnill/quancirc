import numpy as np
from rich.console import Console
from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from qcp_program import Gate, Qubit, Density, Measure, Circuit, Grover


class print_array:

    prec = 3
    
    if isinstance(self, Measure):
        return self.format_measure()
    elif isinstance(self, Gate):
        return self.format_gate()
    elif isinstance(self, Grover):
        return self.format_grover()
    elif isinstance(self, Qubit):
        return self.format_qubit()
    elif isinstance(self, np.ndarray):
        return self.format_ndarray()
    return self.format_default()

class print_array:    #made to try to make matrices look prettier
    """Custom print function to neatly arrange matrices and also print with a nice font"""
    def __init__(self, array):
        self.console = Console()  # Use Rich's Console for rich printing
        self.array = array
        self.prec = 3  # Default precision for numpy formatting
        np.set_printoptions(
            precision=self.prec,
            suppress=True,
            floatmode="fixed")
        if isinstance(array, Measure):
            if array.measure_type == "projective":
                ket_mat = range(array.dim)
                num_bits = int(np.ceil(np.log2(array.dim)))
                np.set_printoptions(linewidth=(10))
                console.print(f"{array.name}",markup=True, style="measure")
                for ket_val, prob_val in zip(ket_mat,array.list_probs()):       #creates a bra ket notation of the given values
                    console.print(f"|{bin(ket_val)[2:].zfill(num_bits)}>  {100*prob_val:.{self.prec}f}%",markup=True, style="measure")
            else:
                console.print(array,markup=True,style="measure")
        elif isinstance(array, Gate):
            if array.dim < 5:
                np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * array.dim, precision=self.prec)
            elif array.dim < 9:
                self.prec = self.prec - 1
                np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * array.dim, precision=self.prec)
            else:
                self.prec = self.prec - 2
                np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * array.dim, precision=self.prec)
            if isinstance(array, Density):
                console.print(array,markup=True,style="density")
            else:
                console.print(array,markup=True,style="gate")
        elif isinstance(array, Grover):
            console.print(array,markup=True, style="prob_dist")
        elif isinstance(array, Qubit):
            np.set_printoptions(linewidth=(10))
            console.print(array,markup=True,style="qubit")
        elif isinstance(array, np.ndarray):
            length = len(array)
            if length < 17:
                np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * np.sqrt(length))
            elif length < 65:
                self.prec = self.prec - 1
                np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * np.sqrt(length), precision=self.prec)
            else:
                self.prec = self.prec - 2
                np.set_printoptions(linewidth=3 + (8 + 2 * self.prec) * np.sqrt(length), precision=self.prec)
            console.print(array, markup=True, style="gate")
        else:
            console.print(array,markup=True,style="info")