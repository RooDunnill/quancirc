import time
start = time.time()
import numpy as np                                              #mostly used to make 1D arrays
import random as rm                                             #used for measuring
import atexit
import matplotlib.pyplot as plt
from rich.console import Console 
from rich.theme import Theme
from rich.table import Table
import cProfile

custom_theme = Theme({"Qubit_style":"spring_green4",
                      "Prob_dist_style":"green4",
                      "Gate_style":"dark_green",
                      "Density_style":"chartreuse4",
                      "info":"grey62",
                      "error":"dark_orange",
                      "measure":"green1",
                      "headers":"dark_goldenrod"})
console = Console(style="none",theme=custom_theme, highlight=False)

def prog_end():    #made it to make the code at the end of the program a little neater
    stop = time.time()
    interval: float = stop - start
    console.print(f"{interval:.3f} seconds elapsed",style="info")
    plt.show()
atexit.register(prog_end)

console.rule(style="headers")
console.rule(f"Quantum Computer Simulator", style="headers")
console.rule(style="headers")
console.print("""Welcome to my Quantum Computer Simulator,
here you can simulate a circuit with any amount of gates and qubits.
You can define your own algorithm in a function and also 
define any gate with the universal gate class. 
The current convention of this program, is that the "first"
gate to be multiplied is at the bottom in a Quantum Circuit.
Now printing the values of the computation:""",style="info")
print("\n \n")


    

class qc_dat:                    #defines a class to store variables in to recall from so that its all
    q0_matrix = [1,0]
    q1_matrix = [0,1]
    qplus_matrix = [1,1]
    qminus_matrix = [1,-1]
    C_Not_info = """This gate is used to change the behaviour of one qubit based on another. 
    This sepecific function is mostly obselete now, it is preferred to use the C Gate class instead"""       #C_Not is mostly obsolete due to new C Gate class       #in one neat area
    C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]#2 Qubit CNot gate in one configuration, need to add more
    X_Gate_info = "Used to flip the Qubit in the X basis. Often seen in the CNot gate."             
    X_matrix = [0,1,1,0]                           
    Y_Gate_info = "Used to flip the Qubit in the Y basis."
    Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]           
    Z_Gate_info = "Used to flip the Qubit in the Z basis. This gate flips from 1 to 0 in the computational basis."
    Z_matrix = [1,0,0,-1]                                      
    n = 1/np.sqrt(2)
    Hadamard_info = """The Hadamard gate is one of the most useful gates, used to convert the Qubit from
    the computation basis to the plus, minus basis. When applied twice, it can lead back to its original value/
    acts as an Identity matrix."""
    Hadamard_matrix = np.array([n,n,n,-n])
    U_Gate_info = """This is a gate that can be transformed into most elementary gates using the constants a,b and c.
    For example a Hadamard gate can be defined with a = pi, b = 0 and c = pi while an X Gate can be defined by
     a = pi/2 b = 0 and c = pi. """
    Identity_info = """
    Identity Matrix: This matrix leaves the product invariant after multiplication.
    It is mainly used in this program to increase the dimension
    of other matrices. This is used within the tensor products when
    a Qubit has no gate action, but the others do."""
    Identity_matrix = [1,0,0,1]
    error_class = "the operation is not with the correct class"
    error_mat_dim = "the dimensions of the matrices do not share the same value"
    error_value = "the value selected is outside the correct range of options"
    error_qubit_num = "you can't select the same qubit for both inputs of the operation gate and control gate."
    error_qubit_pos = "one of the selected qubits must be 1 which represents the top left of the matrix rather than qubit 1."
    Density_matrix_info = "test"
    prob_dist_info = "this is a matrix of the probability of each measurement occuring within a group of qubits."
    error_trace = "the trace does not equal 1 and so the calculation has gone wrong somewhere."
    error_imag_prob = "the probability must be all real values."
    error_norm = "the sum of these values must equal 1 to preserve probability."
    qubit_info = """The Qubit is the quantum equivalent to the bit. However, due to the nature of 
        Quantum Mechanics, it can take any value rather than just two. However, by measuring the state
        in which it is in, you collapse the wavefunction and the Qubit becomes 1 of two values, 1 or 0."""
    gate_info = """Gates are used to apply an operation to a Qubit. 
    They are normally situated on a grid of n Qubits.
    Using tensor products, we can combine all the gates 
    at one time instance together to create one unitary matrix.
    Then we can matrix multiply successive gates together to creat one
    universal matrix that we can apply to the Qubit before measuring"""
    
class QC_error(Exception):
    """Creates my own custom errors defined in qc_dat."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

def trace(matrix):
    """Just computes the trace of a matrix, mostly used as a checker"""
    if isinstance(matrix, Gate):
        tr = 0
        for i in range(matrix.dim):
            tr += matrix.matrix[i+i*matrix.dim]
        return tr
    else:
        raise QC_error(qc_dat.error_class)

def is_real(obj):
    if isinstance(obj, complex):
        if np.imag(obj) < 1e-5:
            return True
    elif isinstance(obj, (int, float)):
        return True
    else:
        return False


class Qubit:              
    def __init__(self, name, vector) -> None:
        self.name = name
        self.vector = np.array(vector,dtype=np.complex128)
        self.dim = len(vector)                    #used constantly in all calcs so defined it universally
        self.shift = self.dim.bit_length() - 1
        
    def __str__(self):
        return f"{self.name}\n{self.vector}"   #did this so that the matrix prints neatly
    
    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.vector}[/not bold]"
    
    def __matmul__(self, other):               #this is an n x n tensor product function
        if isinstance(other, Qubit):           #although this tensors are all 1D   
            self_name_size = int(np.log2(self.dim))
            other_name_size = int(np.log2(other.dim))
            new_name = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>" 
            new_length: int = self.dim*other.dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            other_shift = other.dim.bit_length() - 1
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other.dim):          #iterates up and down the second ket
                    new_vector[j+(i << other_shift)] += self.vector[i]*other.vector[j] #adds the values into
            return Qubit(new_name, np.array(new_vector))    #returns a new Object with a new name too
        else:
            raise QC_error(qc_dat.error_class)

    def __ipow__(self, other):                 #denoted **=
        if isinstance(self, Qubit):  
            self = self @ other
            return self
        else:
            raise QC_error(qc_dat.error_class)


    def norm(self):                 #dunno why this is here ngl, just one of the first functions i tried
        normalise = np.sqrt(sum([i*np.conj(i) for i in self.vector]))
        self.vector = self.vector/normalise

    def qubit_info(self):      
        print(qc_dat.qubit_info)

    def density_mat(self):
        new_name =f"Density matrix of qubit {self.name}"
        new_mat = np.zeros(self.dim*self.dim,dtype=np.complex128)
        qubit_conj = np.conj(self.vector)
        for i in range(self.dim):
            for j in range(self.dim):
                new_mat[j+(i << self.shift)] += qubit_conj[i]*self.vector[j]
        den = Density(new_name, qc_dat.Density_matrix_info, new_mat)
        if abs(1 -trace(den)) < 1e-5:
            return den
        else:
            raise QC_error(qc_dat.error_trace)

    def prob_state(self, meas_state=None, final_gate=None) -> float:
        global is_real
        if meas_state:
            if isinstance(self, Qubit) and isinstance(meas_state, Qubit):
                projector = meas_state.density_mat()
                if final_gate:
                    if isinstance(final_gate, Gate):
                        final_state = final_gate * self
                    else:
                        raise QC_error(qc_dat.error_class)
                else:
                    final_state = self
                den = final_state.density_mat()
                probability = trace(projector * den)
                is_real = np.isreal(probability)
                if is_real is True:
                    return probability
                else:
                    if np.imag(probability) < 1e-5:
                        return probability
                    else:
                        raise QC_error(qc_dat.error_imag_prob)
            else:
                raise QC_error(qc_dat.error_class)

    def prob_dist(self, final_gate=None):
        new_mat = np.zeros(self.dim,dtype=np.float64)
        if isinstance(self, Qubit):
            norm = 0
            if final_gate:
                if isinstance(final_gate, Gate):
                    new_name = f"PD for {self.name} applied to Circuit:"
                    new_state = final_gate * self
                    state_conj = np.conj(new_state.vector)
                    for i in range(self.dim):
                        new_mat[i] = (new_state.vector[i]*state_conj[i]).real
                        norm += new_mat[i]
                else:
                    raise QC_error(qc_dat.error_class)
            else:
                qubit_conj = np.conj(self.vector)
                new_name = f"PD for {self.name}"
                for i in range(self.dim):
                    new_mat[i] = (self.vector[i]*qubit_conj[i]).real
                    norm += new_mat[i]
            if np.isclose(norm, 1.0, atol=1e-5):
                return Prob_dist(new_name, qc_dat.prob_dist_info, np.array(new_mat))
            else:
                raise QC_error(qc_dat.error_norm)
        else:
            raise QC_error(qc_dat.error_class)

    def measure(self, final_gate=None):
        if isinstance(self, Qubit):
            sequence = np.arange(0,self.dim)
            if final_gate:
                if isinstance(final_gate, Gate):
                    PD = self.prob_dist(final_gate)
                    measurement = int(rm.choices(sequence, weights=PD.matrix)[0])
                else:
                    raise QC_error(qc_dat.error_class)
            else:
                PD = self.prob_dist()
                measurement = int(rm.choices(sequence, weights=PD.matrix)[0])
            num_bits = int(np.ceil(np.log2(self.dim)))
            measurement = f"Measured the state: |{bin(measurement)[2:].zfill(num_bits)}>"
            return measurement

    def bloch_plot(self):  #turn this into a class soon
        plot_counter = 0
        vals = np.zeros(2,dtype=np.complex128)
        vals[0] = self.vector[0]
        vals[1] = self.vector[1]
        plotted_qubit = Qubit("", vals)
        den_mat = plotted_qubit.density_mat()
        x = 2*np.real(den_mat.matrix[1])
        y = 2*np.imag(den_mat.matrix[2])
        z = den_mat.matrix[0] - den_mat.matrix[3]
        ax = plt.axes(projection="3d")
        ax.quiver(0,0,0,x,y,z)
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x_sp = np.cos(u)*np.sin(v)
        y_sp = np.sin(u)*np.sin(v)
        z_sp = np.cos(v)
        ax.set_xlabel("X_Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Bloch Sphere")
        ax.plot_surface(x_sp, y_sp, z_sp, color="g", alpha=0.3)
        ax.axes.grid(axis="x")
        ax.text(0,0,1,"|0>")
        ax.text(0,0,-1,"|1>")
        ax.text(1,0,0,"|+>")
        ax.text(-1,0,0,"|->")
        ax.text(0,1,0,"|i>")
        ax.text(0,-1,0,"|-i>")
        ax.plot([-1,1],[0,0],color="black")
        ax.plot([0,0],[-1,1],color="black")
        ax.plot([0,0],[-1,1],zdir="y",color="black")
        plot_counter += 1
        
        


q0 = Qubit("|0>",qc_dat.q0_matrix)
q1 = Qubit("|1>",qc_dat.q1_matrix)
qplus = Qubit("|+>",qc_dat.qplus_matrix)
qminus = Qubit("|->",qc_dat.qminus_matrix)
q0.norm()
q1.norm()
qplus.norm()
qminus.norm()

class Gate:
    def __init__(self, name, info, matrix):
        self.name = name
        self.matrix = np.array(matrix,dtype=np.complex128)
        self.info = info
        self.length = len(matrix)          #naming these matrices and qubits vectors was a stupid idea XD
        self.dim = int(np.sqrt(self.length))
        self.shift = self.dim.bit_length() - 1

    def __str__(self):
        return f"{self.name}\n{self.matrix}"

    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.matrix}[/not bold]"
    

    def __matmul__(self, other):
        if isinstance(other, Gate):
            new_info = "This is a tensor product of gates: "f"{self.name}"" and "f"{other.name}"
            new_name = f"{self.name} @ {other.name}"
            new_length = self.length*other.length
            new_mat = np.zeros(new_length,dtype=np.complex128)
            new_shift = (self.dim*other.dim).bit_length() - 1
            comb_shift = other.shift + new_shift
            for m in range(self.dim):
                for i in range(self.dim):
                    for j in range(other.dim):             #4 is 100 2 is 10
                        for k in range(other.dim):   #honestly, this works but is trash and looks like shit
                            new_mat[k+(j << new_shift)+(i << other.shift)+(m << comb_shift)] += self.matrix[i+(m << self.shift)]*other.matrix[k+(j << other.shift)]
            return Gate(new_name, new_info, np.array(new_mat))
        else:
            raise QC_error(qc_dat.error_class)

    def __ipow__(self, other):    #denoted **=
        if isinstance(self, Gate):  
            self = self @ other
            return self
        else:
            raise QC_error(qc_dat.error_class)
        

    def __mul__(self, other):       #matrix multiplication
        _summ = 0
        if isinstance(self, Gate):
            if isinstance(other, Gate):    #however probs completely better way to do this so might scrap at some point
                if self.dim == other.dim:
                    new_info = "This is a matrix multiplication of gates: "f"{self.name}"" and "f"{other.name}"
                    new_name = f"{self.name} * {other.name}"
                    new_mat = np.zeros(self.length,dtype=np.complex128)
                    for i in range(self.dim):
                        for k in range(self.dim):
                            for j in range(self.dim):    #again a mess and done in a different manner to tensor product
                                _summ += (self.matrix[j+(i << other.shift)]*other.matrix[k+(j << self.shift)])
                            new_mat[k+(i << self.shift)] += _summ
                            _summ = 0
                    if isinstance(other, Density):
                        new_info = "This is the density matrix of: "f"{self.name}"" and "f"{other.name}"
                        return Density(new_name, new_info, new_mat)
                    else:
                        return Gate(new_name, new_info, np.array(new_mat))
                else:
                    raise QC_error(qc_dat.error_mat_dim)
            elif isinstance(other, Qubit):  #splits up based on type as this isnt two n x n but rather n x n and n matrix
                if self.dim == other.dim:
                    new_name = f"{self.name}{other.name}"
                    new_mat = np.zeros(self.dim,dtype=np.complex128)
                    for i in range(self.dim):
                            for j in range(self.dim):
                                _summ += (self.matrix[j+(i << other.shift)]*other.vector[j])
                            new_mat[i] += _summ
                            _summ = 0
                    return Qubit(new_name, np.array(new_mat))
                else:
                    raise QC_error(qc_dat.error_mat_dim)
            else:
                raise QC_error(qc_dat.error_class)
        else:
            raise QC_error(qc_dat.error_class)
    
        


    def __add__(self, other):         #direct sum                   
        if isinstance(other, Gate):                   #DONT TOUCH WITH THE BINARY SHIFTS AS THIS ISNT IN POWERS OF 2
            new_info = "This is a direct sum of gates: "f"{self.name}"" and "f"{other.name}"
            new_name = f"{self.name} + {other.name}"
            new_dim = self.dim + other.dim
            new_length = new_dim**2
            new_mat = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):
                for j in range(self.dim):                   #a lot more elegant
                    new_mat[j+new_dim*i] += self.matrix[j+self.dim*i]
            for i in range(other.dim):     #although would be faster if i made a function to apply straight
                for j in range(other.dim):    #to individual qubits instead
                    new_mat[self.dim+j+self.dim*new_dim+new_dim*i] += other.matrix[j+other.dim*i]
            return Gate(new_name, new_info, np.array(new_mat))
        else:
            raise QC_error(qc_dat.error_class)
    
    def __iadd__(self, other):
        if isinstance(other, Gate):
            self = self + other
            return self
        else:
            raise QC_error(qc_dat.error_class)
        
        

    def gate_info(self):
        print(qc_dat.gate_info)
class C_Gate(Gate):
    def __init__(self, name, info, gate_action, qubit1, qubit2):
        self.name = name
        self.info = info
        self.qubit1 = qubit1   #qubit 1 is the control qubit ie the identity matrix
        self.qubit2 = qubit2   #qubit 2 is the gate qubit, ie x for cnot gate
        self.gate_action = gate_action
        qubit_dist = int(qubit1 - qubit2)
        if qubit1 == 1 or qubit2 == 1:
            if qubit_dist < 0:
                i = 0
                Id = Identity
                while i < int(abs(qubit_dist) - 1):
                    Id += Identity
                    i += 1
                new_mat = Id + self.gate_action
            elif qubit_dist > 0:
                i = 0
                Id = self.gate_action
                while i < int(abs(qubit_dist)):
                    Id += Identity
                    i += 1
                new_mat = Id
            else:
                raise QC_error(qc_dat.error_qubit_num)
        else:
            raise QC_error(qc_dat.error_qubit_pos)
        self.matrix = new_mat.matrix
        self.dim = int(abs(qubit_dist)*Identity.dim+gate_action.dim)
        self.length = self.dim*self.dim
        self.shift = self.dim.bit_length() - 1
                

class U_Gate(Gate):
    def __init__(self, name, info, a, b, c):
        self.name = name
        self.info = info
        self.a = a
        self.b = b
        self.c = c
        self.matrix = np.array([[np.cos(self.a/2)],
                               [-np.exp(np.complex128(0-1j)*self.c)*np.sin(self.a/2)],
                               [np.exp(np.complex128(0+1j)*self.b)*np.sin(self.a/2)],
                               [np.exp(np.complex128(0+1j)*(self.b+self.c))*np.cos(self.a/2)]],dtype=np.complex128)
        self.length = len(self.matrix)
        self.dim = int(np.sqrt(self.length))
        self.shift = self.dim.bit_length() - 1
        

class Phase_Gate(Gate):
    def __init__(self, name, info, theta):
        self.name = name
        self.info = info
        self.matrix = np.array([1,0,0,np.exp(np.complex128(0-1j)*theta)])
        self.length = len(self.matrix)
        self.dim = int(np.sqrt(self.length))
        self.shift = self.dim.bit_length() - 1

class Density(Gate):
    def __init__(self, name, info, matrix):
        self.name = name
        self.info = info
        self.matrix = matrix
        self.length = len(self.matrix)
        self.dim = int(np.sqrt(self.length))
        self.shift = self.dim.bit_length() - 1

    def __str__(self):
        return f"{self.name}\n{self.matrix}"
    
    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.matrix}[/not bold]"
    
    def __and__(self, other):
        new_name = f"{self.name} + {other.name}"
        new_mat = np.zeros(self.length,dtype=np.complex128)
        if isinstance(self, Density) and isinstance(other, Density):
            for i in range(self.length):
                new_mat[i] = self.matrix[i] + other.matrix[i]
            return Density(new_name, qc_dat.Density_matrix_info, np.array(new_mat))
        else:
            raise QC_error(qc_dat.error_class)

class Prob_dist(Qubit):
    def __init__(self, name, info, matrix):
        self.name = name
        self.info = info
        self.matrix = matrix
        self.dim = len(self.matrix)
    
    def __str__(self):
        return f"{self.name}\n{self.matrix}"
    
    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.matrix}[/not bold]"

        

class print_array:    #made to try to make matrices look prettier
    def __init__(self, array):
        self.console = Console()  # Use Rich's Console for rich printing
        self.array = array
        self.prec = 3  # Default precision for numpy formatting
        np.set_printoptions(
            precision=self.prec,
            suppress=True,
            floatmode="fixed")
        
        if isinstance(array, Density):
            if array.dim < 9:
                np.set_printoptions(linewidth=(3 + 2 * (3 + self.prec)) * array.dim)
            else:
                np.set_printoptions(linewidth=(3 + 2 * (4 + self.prec)) * array.dim)
            console.print(array,markup=True,style="Density_style")
        elif isinstance(array, Prob_dist):
            ket_mat = np.arange(0,array.dim)
            num_bits = int(np.ceil(np.log2(array.dim)))
            np.set_printoptions(linewidth=(10))
            console.print(f"{array.name}",markup=True, style="Prob_dist_style")
            for ket_val, prob_val in zip(ket_mat,array.matrix):
                console.print(f"|{bin(ket_val)[2:].zfill(num_bits)}>  {prob_val:.{3}f}",markup=True, style="Prob_dist_style")
        elif isinstance(array, Qubit):
            np.set_printoptions(linewidth=(10))
            console.print(array,markup=True,style="Qubit_style")
        elif isinstance(array, Gate):
            if array.dim < 9:
                np.set_printoptions(linewidth=(3 + 2 * (3 + self.prec)) * array.dim)
            else:
                np.set_printoptions(linewidth=(3 + 2 * (4 + self.prec)) * array.dim)
            console.print(array,markup=True,style="Gate_style")
        
        else:
            console.print(array,markup=True,style="White")

X_Gate = Gate("X", qc_dat.X_Gate_info, qc_dat.X_matrix)
Y_Gate = Gate("Y",qc_dat.Y_Gate_info, qc_dat.Y_matrix)
Z_Gate = Gate("Z",qc_dat.Z_Gate_info, qc_dat.Z_matrix)
Identity = Gate("I",qc_dat.Identity_info, qc_dat.Identity_matrix)
Hadamard = Gate("H",qc_dat.Hadamard_info, qc_dat.Hadamard_matrix)
U_Gate_X = U_Gate("Universal X", qc_dat.U_Gate_info, np.pi, 0, np.pi)
U_Gate_H = U_Gate("Universal H", qc_dat.U_Gate_info, np.pi/2, 0, np.pi)
CNot_flip = C_Gate("CNot_flip", qc_dat.C_Not_matrix, X_Gate, 2, 1)
CNot = C_Gate("CNot", qc_dat.C_Not_matrix, X_Gate, 1, 2)
gate_operations = {
    "H": Hadamard,
    "X": X_Gate,
    "Y": Y_Gate,
    "Z": Z_Gate,
    "I": Identity
}
qub = q1 @ q1 @ q1
def fractional_binary(qub,m):
    num_bits = int(np.ceil(np.log2(qub.dim)))
    x_vals = qub.name[1:1+num_bits]
    frac_bin = ("0." + x_vals)
    val = 0
    if m <= num_bits:
        for i in range(m):
            val += float(frac_bin[i+2])*2**-(i+1)
            print(float(frac_bin[i+2]))
        print("new func")
        return val

def quant_fourier_trans(qub):
    old_name = qub.name
    n = int(np.ceil(np.log2(qub.dim)))
    frac_init = fractional_binary(qub,1)
    four_qub_init = Qubit("",np.array([1,np.exp(2*1j*np.pi*frac_init)]))
    print_array(four_qub_init)
    four_qub_init.norm()
    four_qub_sum = four_qub_init
    for j in range(n-1):
        frac = fractional_binary(qub,j+2)
        four_qub = Qubit("",np.array([1,np.exp(2*1j*np.pi*frac)]))
        four_qub.norm()
        print_array(four_qub)
        four_qub_sum **= four_qub
    four_qub_sum.name = f"QFT of {old_name}"
    return four_qub_sum

def alg_template(Qubit):         #make sure to mat mult the correct order
    circuit = [["X","H","X"],
               ["H","X","H"],
               ["Z","Z","H"]]
    console.rule(f"Algorithm acting on {Qubit.name} state", style="headers")
    gate1 = CNot
    gate2 = CNot_flip
    gate3 = CNot
    alg = gate3 * gate2 * gate1
    print_array(alg)
    _pd_result = Qubit.prob_dist(alg)
    result = Qubit.measure(alg)
    print_array(_pd_result)
    print_array(result)
qub = q0 @ q0
alg_template(qub)

def alg_template2(Qubit):         #make sure to mat mult the correct order
    circuit = [["X","H","X"],
               ["H","X","H"],
               ["Z","Z","H"]]
    console.rule(f"Algorithm acting on {Qubit.name} state", style="headers")
    gate1 = X_Gate @ X_Gate @ X_Gate
    alg = gate1
    _pd_result = Qubit.prob_dist(alg)
    result = Qubit.measure(alg)
    print_array(_pd_result)
    print_array(result)
qub = q0 @ q0 @ q1
#alg_template2(qub)
oracle_values = [3, 1, 6]


def sort(obj, oracle_values=None):
    if isinstance(obj, Prob_dist):
        save_probs = np.zeros(len(oracle_values))
        k = 0
        print_array(obj)
        for i in obj.matrix:
            for j in oracle_values:
                if i < obj.matrix[j]:
                    print_array(f"One of the oracle values is: {j}")
                    save_probs[k] = j
                    k += 1
                elif i > obj.matrix[j]:
                    print_array(f"One of the oracle values may be {i}")
                else:
                    print_array(f"Oops")
        return save_probs
    else:
        raise QC_error(qc_dat.error_class)


def phase_oracle(qub, oracle_values):
    flip = np.ones(qub.dim)
    for i in oracle_values:
        flip[i] = flip[i] - 2
    for j, vals in enumerate(flip):
        qub.vector[j] = qub.vector[j] * vals        
    return qub

def grover_alg(oracle_values, n, iterations=None):
    search_space: int = 2**n
    console.rule(f"Grovers algorithm with values:{oracle_values} and a search space of {search_space}", style="headers")
    op_iter = (np.pi/4)*np.sqrt((search_space)/len(oracle_values)) - 0.5
    if iterations == None:
        iterations = op_iter
        if iterations < 1:
            iterations = 1.0000
        print_array(f"Optimal amount of iterations are: {iterations:.4}")
    qub = q0
    had_op = Hadamard
    flip_cond = - np.ones(qub.dim**n)
    flip_cond[0] += 2
    for i in range(n-1):
        qub **= q0
        had_op **= Hadamard
    it = 0
    while it < int(iterations):
        if it != 0:
            qub = final_state
        initialized_qubit = had_op * qub
        intermidary_qubit = had_op * phase_oracle(initialized_qubit, oracle_values)
        for j, vals in enumerate(flip_cond):
            intermidary_qubit.vector[j] = intermidary_qubit.vector[j] * vals 
        final_state = had_op * intermidary_qubit
        it += 1
        print_array(f"Iteration number: {it}")
        final_state.name = f"Grover Search with Oracle Values {oracle_values}, after {int(iterations)} iterations is: "
    final_state = final_state.prob_dist()
    print_array(final_state)
    console.rule(f"", style="headers")
    return final_state, op_iter
q00 = q0 @ q0 
q01 = q0 @ q1
print_array(qplus.bloch_plot())
cProfile.run("grover_alg(oracle_values, 4)")
