import time
start = time.time()
import numpy as np                                              #mostly used to make 1D arrays
from random import choices, randint                                            #used for measuring
import atexit
import matplotlib.pyplot as plt
from rich.console import Console
from rich.theme import Theme
from scipy.linalg import sqrtm, logm
import cProfile


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


class Timer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time

    def elapsed(self):
        current_time = time.perf_counter()
        interval_time = current_time - self.last_time 
        total_time = current_time - self.start_time 
        self.last_time = current_time
        return interval_time, total_time
    
def prog_end():    #made it to make the code at the end of the program a little neater
    if __name__ == "__main__":
        main()
    stop = time.time()
    interval: float = stop - start
    console.rule(f"{interval:.3f} seconds elapsed",style="main_header")
    plt.show()
atexit.register(prog_end)

console.rule(style="main_header")     #creates the start message
console.rule(f"Quantum Computer Simulator", style="main_header", characters="#")
console.rule(style="main_header")
console.print("""Welcome to our Quantum Computer Simulator,
here you can simulate a circuit with any amount of gates and qubits.
You can define your own algorithm in a function and also 
define any gate with the universal gate class. 
The current convention of this program, is that the "first"
gate to be multiplied is at the bottom in a Quantum Circuit.
Now printing the values of the computation:""",style="info")
print("\n \n")

class qc_dat:                    #defines a class to store variables in to recall from so that its all
    C_Not_info = """This gate is used to change the behaviour of one qubit based on another. 
    This sepecific function is mostly obselete now, it is preferred to use the C Gate class instead"""       #C_Not is mostly obsolete due to new C Gate class       #in one neat area
    C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]#2 Qubit CNot gate in one configuration, need to add more
    X_Gate_info = "Used to flip the Qubit in the X basis. Often seen in the CNot gate."                             
    Y_Gate_info = "Used to flip the Qubit in the Y basis."       
    Z_Gate_info = "Used to flip the Qubit in the Z basis. This gate flips from 1 to 0 in the computational basis."                           
    Hadamard_info = """The Hadamard gate is one of the most useful gates, used to convert the Qubit from
    the computation basis to the plus, minus basis. When applied twice, it can lead back to its original value/
    acts as an Identity matrix."""
    U_Gate_info = """This is a gate that can be transformed into most elementary gates using the constants a,b and c.
    For example a Hadamard gate can be defined with a = pi, b = 0 and c = pi while an X Gate can be defined by
     a = pi/2 b = 0 and c = pi. """
    Identity_info = """
    Identity Matrix: This matrix leaves the product invariant after multiplication.
    It is mainly used in this program to increase the dimension
    of other matrices. This is used within the tensor products when
    a Qubit has no gate action, but the others do."""
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
    error_iterations = "to customise the iterations value, you must provide the number of qubits used in the search"
    qubit_info = """The Qubit is the quantum equivalent to the bit. However, due to the nature of 
    Quantum Mechanics, it can take any value rather than just two. However, by measuring the state
    in which it is in, you collapse the wavefunction and the Qubit becomes 1 of two values, 1 or 0."""
    gate_info = """Gates are used to apply an operation to a Qubit. 
    They are normally situated on a grid of n Qubits.
    Using tensor products, we can combine all the gates 
    at one time instance together to create one unitary matrix.
    Then we can matrix multiply successive gates together to creat one
    universal matrix that we can apply to the Qubit before measuring"""
    error_mixed_state = "the mixed states must each have their own probability values"
    error_density_vectors = "either the vectors are the wrong data type, or no vector has been provided to make a density matrix of"
    error_kwargs = "not enough key word arguments provided"
    error_empty_circuit = "the circuit is empty and needs atleast one gate to work"

class QC_error(Exception):                 #a custom error class to raise custom errors from qc_dat
    """Creates my own custom errors defined in qc_dat."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"
 
def trace(matrix) -> float:                #calculates the trace of a matrix, either for a gate class or for a normal array
    """Just computes the trace of a matrix, mostly used as a checker"""
    if isinstance(matrix, Gate):
        tr = 0
        for i in range(matrix.dim):
            tr += matrix.matrix[i+i*matrix.dim]
        return tr
    elif isinstance(matrix, np.ndarray):
        dim = int(np.sqrt(len(matrix)))
        tr = 0
        for i in range(dim):
            tr += matrix[i+i*dim]
        return tr
    else:
        raise QC_error(qc_dat.error_class)
    
def reshape_matrix(matrix: np.ndarray) -> np.ndarray:
    length = len(matrix)
    dim = int(np.sqrt(length))
    if dim**2 != length:
        QC_error(f"The matrix cannot be reshaped into a perfect square")
    reshaped_matrix = []
    for i in range(dim):
        row = matrix[i * dim : (i + 1) * dim]
        reshaped_matrix.append(row)
    return np.array(reshaped_matrix)

def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    dim = len(matrix)
    length = dim**2
    flattened_matrix = np.zeros(length, dtype=np.complex128)
    for j in range(dim):
        for i in range(dim):
            flattened_matrix[j * dim + i] += matrix[j][i]
    return flattened_matrix
        
def is_real(obj):                 #pretty irrelevant but is used for checking probs are real
    if isinstance(obj, complex):
        if np.imag(obj) < 1e-5:
            return True
    elif isinstance(obj, (int, float)):
        return True
    else:
        return False

def comp_Grover_test(n, **kwargs):
        g_loops = kwargs.get("g_loops", 10)
        warmup_timer = Timer()
        warmup = warmup_timer.elapsed()
        loops = np.arange(2,n+1)
        times_array = np.zeros((n-1, 3))
        for i in loops:
            n=int(i)
            test_timer = Timer()
            for j in range(g_loops): Grover(oracle_value_test, n=n).run()
            time_slow = test_timer.elapsed()[0]
            test_timer = Timer()
            for j in range(g_loops): Grover(oracle_value_test, n=n, fast=True).run()
            time_fast = test_timer.elapsed()[0]
            times_array[i-2] = np.array([n, time_slow/g_loops, time_fast/g_loops])
        print_array(f"Qubits, s time, f time")
        print(times_array)

def time_test(n, fast=True, iterations=None, **kwargs):
    it_type = kwargs.get("it_type", None)
    oracle_value_test = [0]
    g_loops = kwargs.get("g_loops", 10)
    rand = kwargs.get("rand", None)
    if rand:
        oracle_value_test = np.zeros(rand)
        for i in range(rand):
            oracle_value_test[i] = randint(0,2**n - 1)
    warmup_timer = Timer()
    warmup = warmup_timer.elapsed
    loops = np.arange(2,n+1)
    times_array = np.zeros((n-1, 2))
    for i in loops:
        n=int(i)
        test_timer = Timer()
        if it_type and rand:
            Grover(rand, fast=fast).run()
        else:
            for j in range(g_loops): Grover(oracle_value_test, n=n, fast=fast, iterations=iterations).run()
        time = test_timer.elapsed()[0]
        times_array[i-2] = np.array([n, time/g_loops])
    print_array(f"Qubits, time")
    print(times_array)

def top_probs(prob_list: np.ndarray, n: int) -> np.ndarray:             #sorts through the probability distribution and finds the top n probabilities corresponding to the length n or the oracle values
        top_n = np.array([], dtype=prob_list.dtype)
        temp_lst = prob_list.copy()  
        for _ in range(n):
            max_value = np.max(temp_lst)
            top_n = np.append(top_n, max_value)
            temp_lst = np.delete(temp_lst, np.argmax(temp_lst))
        result = []
        used_count = {} 
        for i, num in enumerate(prob_list):
            if num in top_n and used_count.get(num, 0) < np.count_nonzero(top_n == num):        #this accounts for if you have two numbers with the same value
                result.append((i, num))
                used_count[num] = used_count.get(num, 0) + 1
        return np.array(result, dtype=object)

def binary_entropy(prob: float) -> float:
    if isinstance(prob, (float, int)):
        if int(prob) ==  0 or int(prob) == 1:
            return 0.0
        else:
            return -prob*np.log2(prob) - (1 - prob)*np.log2(1 - prob)
    else:
        raise QC_error(f"Binary value must be a float")


class Qubit:                                           #creates the qubit class
    def __init__(self, **kwargs) -> None:
        self.state_type: str = kwargs.get("type", "pure")                   #the default qubit is a single pure qubit |0>
        self.name: str = kwargs.get("name","|Quantum State>")
        self.vector = np.array(kwargs.get("vector",np.array([1,0])),dtype=np.complex128)
        if self.vector.ndim == 1:
            self.dim: int = len(self.vector)                    #used constantly in all calcs so defined it universally
        else:
            self.dim: int = len(self.vector[0])
        self.n: int = int(np.log2(self.dim))
        if "vectors" in kwargs:
            if self.state_type == "mixed":
                    self.build_mixed_state(kwargs)
            elif self.state_type == "seperable":
                self.build_seperable_state(kwargs)

    def build_mixed_state(self, kwargs):
        if isinstance(kwargs.get("vectors")[0], Qubit):
            self.vector = np.array(kwargs.get("vectors",[]))
        else:
            self.vector = np.array(kwargs.get("vectors",[]),dtype=np.complex128)
        self.weights = kwargs.get("weights", [])
        if len(self.weights) != len(self.vector):
            raise QC_error(qc_dat.errror_mixed_state)
            
    def build_seperable_state(self, kwargs):
        qubit_states = np.array(kwargs.get("vectors",[]))
        if isinstance(qubit_states[0], np.ndarray):                         #creates the vector for the seperable states for custom vector states
            qubit_states = np.array(kwargs.get("vectors",[]), dtype = np.complex128)
            self.vector = qubit_states[0]
            for state in qubit_states[1:]:
                self.vector = self @ state
            self.name = f"Seperable state of custom states"
        elif isinstance(qubit_states[0], Qubit):                          #creates a seperable state for the tensor of qubits together
            self.vector = qubit_states[0].vector
            state_name_size = int(np.log2(qubit_states[0].dim))
            self.name = qubit_states[0].name                   #name doesnt work atm
            for state in qubit_states[1:]:
                self_name_size = int(np.log2(self.dim))
                self.vector = self @ state.vector
                self.name = f"|{self.name[1:self_name_size+1]}{state.name[1:state_name_size+1]}>"
        else:
            raise QC_error(qc_dat.error_class)

    @classmethod                 #creates the default qubits, using class methods allows you to define an instance within the class
    def q0(cls, **kwargs):
        n = kwargs.get("n", 1)
        q0_vector = [1,0]
        q0_name = f"|0>"
        if n != 1:
            q0_vector = np.zeros(2**n, dtype=np.complex128)
            q0_vector[0] = 1
            zeros_str = "0" * n
            q0_name = f"|{zeros_str}>"
        return cls(name=q0_name, vector=q0_vector)

    @classmethod
    def q1(cls, **kwargs):
        n = kwargs.get("n", 1)
        q1_vector = [0,1]
        q1_name = f"|1>"
        if n != 1:
            q1_vector = np.zeros(2**n, dtype=np.complex128)
            q1_vector[-1] = 1
            ones_str = "1" * n
            q1_name = f"|{ones_str}>"
        return cls(name=q1_name, vector=q1_vector)

    @classmethod
    def qp(cls):
        n = 1/np.sqrt(2)
        qp_vector = [n,n]
        return cls(name="|+>", vector=qp_vector)

    @classmethod
    def qm(cls):
        n = 1/np.sqrt(2)
        qm_vector = [n,-n]
        return cls(name="|->", vector=qm_vector)
    
    @classmethod
    def qpi(cls):
        n =1/np.sqrt(2)
        qpi_vector = np.array([n+0j,0+n*1j],dtype=np.complex128)
        return cls(name="|i>", vector=qpi_vector)
    
    @classmethod
    def qmi(cls):
        n =1/np.sqrt(2)
        qmi_vector = np.array([n+0j,0-n*1j],dtype=np.complex128)
        return cls(name="|-i>", vector=qmi_vector)

    
    def __rich__(self):
            if self.state_type == "mixed":
                return f"[bold]{self.name}[/bold]\n[not bold]Vectors:\n {self.vector}[/not bold]\n and Weights:\n {self.weights}"
            return f"[bold]{self.name}[/bold]\n[not bold]{self.vector}[/not bold]"
    
    def __matmul__(self, other):               #this is an n x n tensor product function
        if isinstance(other, Qubit):           #although this tensors are all 1D  
            self_name_size = int(np.log2(self.dim))
            other_name_size = int(np.log2(other.dim)) 
            new_name = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>"
            new_length: int = self.dim*other.dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other.dim):          #iterates up and down the second ket
                    new_vector[j+(i * other.dim)] += self.vector[i]*other.vector[j] #adds the values into each element of the vector
            return Qubit(name=new_name, vector=new_vector, type=self.state_type)    #returns a new Qubit instance with a new name
        elif isinstance(other, np.ndarray):                 #used for when you just need to compute it for a given array, this is for creating seperable states
            other_dim = len(other)
            new_length: int = self.dim*other_dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other_dim):          #iterates up and down the second ket
                    new_vector[j+(i * other_dim)] += self.vector[i]*other[j] #adds the values into
            self.dim = new_length
            self.vector = new_vector
            return self.vector    #returns a new Object with a new name too
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

    def bloch_plot(self):  #turn this into a class soon, pretty useless but might be worth for report or presentation
        plot_counter = 0
        vals = np.zeros(2,dtype=np.complex128)
        vals[0] = self.vector[0]
        vals[1] = self.vector[1]
        plotted_qubit = Qubit(vector=vals)
        den = Density(qubit=plotted_qubit)
        den_mat = den.rho
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
        
q0 = Qubit.q0()
q1 = Qubit.q1()
qp = Qubit.qp()
qm = Qubit.qm()


class Gate:            #creates a gate class to enable unique properties
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", None)
        self.info = kwargs.get("info", None)
        self.matrix = np.array(kwargs.get("matrix", None),dtype=np.complex128)
        if self.matrix is None:
            raise QC_error(qc_dat.error_kwargs)
        self.length: int = len(self.matrix)
        self.dim: int = int(np.sqrt(self.length))
        self.n: int =  0 if self.dim == 0 else int(np.log2(self.dim))

    @classmethod                #again creates some of the default gates, for ease of use and neatness
    def X_Gate(cls):
        X_matrix = [0,1,1,0]
        return cls(name="X Gate", matrix=X_matrix,info=qc_dat.X_Gate_info)
    
    @classmethod
    def Y_Gate(cls):
        Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]
        return cls(name="Y Gate", matrix=Y_matrix,info=qc_dat.Y_Gate_info)

    @classmethod
    def Z_Gate(cls):
        Z_matrix = [1,0,0,-1]
        return cls(name="Z Gate", matrix=Z_matrix, info=qc_dat.Z_Gate_info)

    @classmethod
    def Identity(cls, **kwargs):                     #eventually want to make this so its n dimensional
        n = kwargs.get("n", 1)
        if isinstance(n, int):
            if n == 0:
                new_mat = np.array([1], dtype=np.complex128)
                return cls(name="Identity Gate", matrix=new_mat, info=qc_dat.Identity_info, dim=0)

            dim = int(2**n)
            new_mat = np.zeros(dim**2, dtype=np.complex128)
            for i in range(dim):
                new_mat[i+ dim * i] += 1
            return cls(name="Identity Gate", matrix=new_mat, info=qc_dat.Identity_info)
        else: 
            Id_matrix = [1,0,0,1]
            return cls(name="Identity Gate", matrix=Id_matrix, info=qc_dat.Identity_info)
    
    @classmethod          
    def Hadamard(cls):
        n = 1/np.sqrt(2)
        H_matrix = [n,n,n,-n]
        return cls(name="Hadamard", matrix=H_matrix, info=qc_dat.Hadamard_info)

    @classmethod                                        #allows for the making of any phase gate of 2 dimensions
    def P_Gate(cls, theta):
        P_matrix = [1,0,0,np.exp(np.complex128(0-1j)*theta)]
        return cls(name=f"Phase Gate with a phase {theta:.3f}", matrix=P_matrix)

    @classmethod                                #allows for any unitary gates with three given variables
    def U_Gate(cls, a, b, c):
        U_matrix = [np.cos(a/2),
                    -np.exp(np.complex128(0-1j)*c)*np.sin(a/2),
                    np.exp(np.complex128(0+1j)*b)*np.sin(a/2),
                    np.exp(np.complex128(0+1j)*(b+c))*np.cos(a/2)]
        return cls(name=f"Unitary Gate with values (a:{a:.3f}, b:{b:.3f}, c:{c:.3f})", matrix=U_matrix)

    @classmethod                             #creates any specific control gate
    def C_Gate(cls, **kwargs):
        gate_type: str = kwargs.get("type", "standard")
        gate_action: Gate = kwargs.get("gate", X_Gate)
        new_gate: Gate = Identity & gate_action
        if gate_type == "standard":
            return cls(name=f"Control {gate_action.name}", matrix=new_gate.matrix)
        elif gate_type == "inverted":
            new_gate = Gate.Swap() * new_gate * Gate.Swap()
            return cls(name=f"Inverted Control {gate_action.name}", matrix=new_gate.matrix)

    @classmethod
    def Swap(cls, **kwargs):
        n_is_2 = [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1]
        n: int = kwargs.get("n", 2)
        if n == 2:
            return cls(name=f"2 Qubit Swap gate", matrix=n_is_2)

    def __str__(self):
        return f"{self.name}\n{self.matrix}"

    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.matrix}[/not bold]"
    
    def __matmul__(self, other: "Gate") -> "Gate":      #adopts the matmul notation to make an easy tensor product of two square matrices
        if isinstance(other, Gate):
            new_info: str = "This is a tensor product of gates: "f"{self.name}"" and "f"{other.name}"
            new_name: str = f"{self.name} @ {other.name}"
            new_length: int = self.length*other.length
            new_mat = np.zeros(new_length,dtype=np.complex128)
            new_dim: int = self.dim * other.dim
            for m in range(self.dim):
                for i in range(self.dim):
                    for j in range(other.dim):             #4 is 100 2 is 10
                        for k in range(other.dim):   #honestly, this works but is trash and looks like shit
                            index = k+j*new_dim+other.dim*i+other.dim*new_dim*m
                            new_mat[index] += self.matrix[i+self.dim*m]*other.matrix[k+other.dim*j]
            return Gate(name=new_name, info=new_info, matrix=new_mat)
        else:
            raise QC_error(qc_dat.error_class)

    def __ipow__(self, other: "Gate") -> "Gate":    #denoted **=
        if isinstance(self, Gate):  
            self = self @ other
            return self
        else:
            raise QC_error(qc_dat.error_class)
        
    def FWHT(self, other):
        if isinstance(other, Qubit):
            sqrt2_inv = 1/np.sqrt(2)
            vec = other.vector
            for i in range(other.n):                                            #loops through each size of qubit below the size of the state
                step_size = 2**(i + 1)                                          #is the dim of the current qubit tieration size 
                half_step = step_size // 2                                      #half the step size to go between odd indexes
                outer_range = np.arange(0, other.dim, step_size)[:, None]       #more efficient intergration of a loop over the state dim in steps of the current dim 
                inner_range = np.arange(half_step)                               
                indices = outer_range + inner_range                        
                a, b = vec[indices], vec[indices + half_step]
                vec[indices] = (a + b) * sqrt2_inv
                vec[indices + half_step] = (a - b) * sqrt2_inv                            #normalisation has been taken out giving a slight speed up in performance
            return other
        else:
            raise TypeError(f"This can't act on this type, only on Qubits")

    def fractional_binary(qub,m):             #for shors
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


    def QFT(self, other):          #also for shors although used in other algorithms
        old_name = other.name
        n = int(np.ceil(np.log2(other.dim)))
        frac_init = fractional_binary(other,1)
        four_qub_init = Qubit(vector=np.array([1,np.exp(2*1j*np.pi*frac_init)]))
        four_qub_init.norm()
        four_qub_sum = four_qub_init
        for j in range(n-1):
            frac = fractional_binary(other,j+2)
            four_qub = Qubit(vector=np.array([1,np.exp(2*1j*np.pi*frac)]))
            four_qub.norm()
            four_qub_sum **= four_qub
        four_qub_sum.name = f"QFT of {old_name}"
        return four_qub_sum

    @staticmethod
    def mul_flat(first, second):
        new_mat = np.zeros(len(first),dtype=np.complex128)
        dim = int(np.sqrt(len(first)))
        dim_range = np.arange(dim)
        for i in range(dim):
            for k in range(dim):
                new_mat[k+(i * dim)] = np.sum(first[dim_range+(i * dim)]*second[k+(dim_range* dim)])
        return new_mat

    def __mul__(self, other):       #matrix multiplication
        if isinstance(self, FWHT):
            return self.FWHT(other)
        elif isinstance(self, QFT):
            return self.QFT(other)
        elif isinstance(self, Gate):
            if isinstance(other, Qubit):  #splits up based on type as this isnt two n x n but rather n x n and n matrix
                if self.dim == other.dim:
                    new_mat = np.zeros(self.dim,dtype=np.complex128)
                    for i in range(self.dim):
                        row = self.matrix[i * self.dim:(i + 1) * self.dim]
                        new_mat[i] = np.sum(row[:] * other.vector[:])
                    other.vector = new_mat
                    other.name = self.name + other.name
                    return other
                else:
                    raise QC_error(qc_dat.error_mat_dim)
            elif isinstance(other, Gate):    #however probs completely better way to do this so might scrap at some point
                if self.dim == other.dim:
                    new_mat = Gate.mul_flat(self.matrix, other.matrix)
                    if isinstance(other, Density):
                        new_info = "This is the density matrix of: "f"{self.name}"" and "f"{other.name}"
                        new_name: str = f"{self.name} * {other.name}"
                        return Density(name=new_name, info=new_info, matrix=np.array(new_mat, dtype=np.complex128))
                    else:
                        self.info: str = "This is a matrix multiplication of gates: "f"{self.name}"" and "f"{other.name}"
                        self.name: str = f"{self.name} * {other.name}"
                        self.matrix = np.array(new_mat, dtype=np.complex128)
                        return self
                else:
                    raise QC_error(qc_dat.error_mat_dim)
            elif isinstance(other, np.ndarray):
                other_dim = int(np.sqrt(len(other)))
                if self.dim == other_dim:
                    new_mat = Gate.mul_flat(self.matrix, other)
                    self.matrix = np.array(new_mat, dtype=np.complex128)
                    return self
            else:
                raise QC_error(qc_dat.error_class)
        elif isinstance(self, np.ndarray):
            if isinstance(other, Gate):
                new_mat = Gate.mul_flat(self, other.matrix)
                other.matrix = new_mat
                return other
        else:
            raise QC_error(qc_dat.error_class)
    
    def __and__(self, other: "Gate") -> "Gate":         #direct sum                   
        if isinstance(other, Gate):                   #DONT TOUCH WITH THE BINARY SHIFTS AS THIS ISNT IN POWERS OF 2
            new_info = "This is a direct sum of gates: "f"{self.name}"" and "f"{other.name}"
            new_name = f"{self.name} + {other.name}"
            new_dim: int = self.dim + other.dim
            new_length: int = new_dim**2
            new_mat = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):
                for j in range(self.dim):                   #a lot more elegant
                    new_mat[j+new_dim*i] += self.matrix[j+self.dim*i]
            for i in range(other.dim):     #although would be faster if i made a function to apply straight
                for j in range(other.dim):    #to individual qubits instead
                    new_mat[self.dim+j+self.dim*new_dim+new_dim*i] += other.matrix[j+other.dim*i]
            return Gate(name=new_name, matrix=np.array(new_mat), info=new_info)
        else:
            raise QC_error(qc_dat.error_class)
    
    def __iadd__(self, other: "Gate") -> "Gate":                                  #used almost exclusively for the CNot gate creator
        if isinstance(other, Gate):
            self = self & other
            return self
        else:
            raise QC_error(qc_dat.error_class)
        
    def gate_info(self):
        print(qc_dat.gate_info)

class FWHT(Gate):
    def __init__(self):
        self.name = f"Fast Walsh Hadamard Transform"
        self.info = f"An efficient way to apply a Hadamard to every qubit"
        self.matrix = np.zeros(4)
        self.length: int = len(self.matrix)
        self.dim: int = int(np.sqrt(self.length))
        self.n: int =  0 if self.dim == 0 else int(np.log2(self.dim))

class QFT(Gate):
    def __init__(self):
        self.name = f"Qunatum Fourier Transform"
        self.info = f"An efficient way to compute the Quantum Fourier transform over all qubits in a state"
        self.matrix = np.zeros(4)
        self.length: int = len(self.matrix)
        self.dim: int = int(np.sqrt(self.length))
        self.n: int =  0 if self.dim == 0 else int(np.log2(self.dim))


class Density(Gate):       #makes a matrix of the probabilities, useful for entangled states
    def __init__(self, **kwargs):
        self.state = kwargs.get("state", None)
        if self.state is None:
            self.state_type = kwargs.get("type", None)
            self.name: str = kwargs.get("name", "Density Matrix:")
        elif isinstance(self.state, Qubit):
            self.state_type: str = kwargs.get("type", self.state.state_type)
            self.q_vector: np.ndarray = self.state.vector
            self.length: int = self.state.dim**2
            self.dim: int = int(np.sqrt(self.length))
            self.n:int = int(np.log2(self.dim))
            if self.state.name is not None:
                    desc_name =f"Density matrix of {self.state_type} qubit {self.state.name}:"
            self.name = kwargs.get("name", desc_name)
        else:
            raise QC_error(qc_dat.error_class)
        self.rho = kwargs.get("rho", None if self.state is None else self.construct_density_matrix(self.state))
        self.length = len(self.rho) if self.state is None else self.state.dim**2
        self.dim = int(np.sqrt(self.length))
        self.n = int(np.log2(self.dim))
        self.state_a = kwargs.get("state_a", None)
        self.state_b = kwargs.get("state_b", None)
        self.rho_a = kwargs.get("rho_a", None if self.state_a is None else self.construct_density_matrix(self.state_a))
        self.rho_b = kwargs.get("rho_b", None if self.state_b is None else self.construct_density_matrix(self.state_b))

        
    def __str__(self):
        return f"{self.name}\n{self.rho}"
    
    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.rho}[/not bold]"
    
    def construct_density_matrix(self, calc_state=None) -> np.ndarray:
        if isinstance(calc_state, Qubit):
            if calc_state.state_type in ["pure", "seperable", "entangled"]:
                return self.generic_density(calc_state)
            elif calc_state.state_type == "mixed":
                return self.mixed_state(calc_state)

        
    def generic_density(self, calc_state: Qubit, **kwargs) -> np.ndarray:       #taken from the old density matrix function
        state_vector = calc_state.vector
        calc_state_dim = len(state_vector)
        rho = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
        qubit_conj = np.conj(state_vector)
        for i in range(calc_state_dim):
            for j in range(calc_state_dim):
                rho[j+(i * calc_state_dim)] += qubit_conj[i]*state_vector[j]
        if abs(1 -trace(rho)) < 1e-5:
            return rho
        else:
            raise QC_error(qc_dat.error_trace)
            
    def mixed_state(self, calc_state: Qubit, **kwargs) -> np.ndarray:
        calc_state_dim = len(calc_state)
        state_vector: np.ndarray = calc_state.vector
        qubit_conj: np.ndarray = np.conj(state_vector)
        if isinstance(state_vector[0], np.ndarray):
            rho = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
            rho_sub = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
            for k in range(len(state_vector)):
                for i in range(calc_state_dim):
                    for j in range(calc_state_dim):
                        rho_sub[j+(i * calc_state_dim)] += qubit_conj[k][i]*state_vector[k][j]
                rho += calc_state.weights[k]*rho_sub
                rho_sub = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
            rho_trace: float = trace(rho)
            if abs(1 - rho_trace) < 1e-5:
                return rho
            else:
                print(f"The calculated trace is {rho_trace}")
                raise QC_error(qc_dat.error_trace)
        elif isinstance(state_vector[0], Qubit):
            rho = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
            rho_sub = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
            for k in range(calc_state_dim):
                den = Density(state=state_vector[k],type="pure")
                rho += den.rho * calc_state.weights[k]
            return rho
        else:
            raise QC_error(qc_dat.error_class)


    def fidelity(self) -> float:
        if isinstance(self.rho_a, np.ndarray) and isinstance(self.rho_b, np.ndarray):
            rho1 = reshape_matrix(self.rho_a)
            rho2 = self.rho_b
            sqrt_rho1: np.ndarray = sqrtm(rho1)
            flat_sqrt_rho1 = flatten_matrix(sqrt_rho1)
            product =  flat_sqrt_rho1 * rho2 * flat_sqrt_rho1
            sqrt_product = sqrtm(reshape_matrix(product))
            flat_sqrt_product = flatten_matrix(sqrt_product)
            mat_trace = trace(flat_sqrt_product)
            mat_trace_conj = np.conj(mat_trace)
            self.fidelity_ab = mat_trace*mat_trace_conj
            return self.fidelity_ab

    def trace_distance(self):
        diff_mat = self.rho_a - self.rho_b
        rho_a_dim = int(np.sqrt(len(self.rho_a)))
        dim_range = np.arange(rho_a_dim)
        self.trace_dist = np.sum(0.5 * np.abs(diff_mat[dim_range + rho_a_dim * dim_range]))
        return self.trace_dist

    def vn_entropy(self, rho: np.ndarray) -> float:
        if isinstance(rho, np.ndarray):
            reshaped_rho = reshape_matrix(rho)
            eigenvalues, eigenvectors = np.linalg.eig(reshaped_rho)
            entropy = 0
            for ev in eigenvalues:
                if ev > 0:
                    entropy -= ev * np.log2(ev)
            if entropy < 1e-10:
                entropy = 0.0
            return entropy
        else:
            raise QC_error(f"No rho matrix provided")
        
    def quantum_conditional_entropy(self, rho=None) -> float:    #rho is the one that goes first in S(A|B)
        if all(isinstance(i, np.ndarray) for i in (self.rho, self.rho_a, self.rho_b)):
            if rho == "rho_a" or "a" or "A":
                cond_ent = self.vn_entropy(self.rho) - self.vn_entropy(self.rho_a)
                return cond_ent
            elif rho == "rh0_b" or "b" or "B":
                cond_ent = self.vn_entropy(self.rho) - self.vn_entropy(self.rho_b)
                return cond_ent
        else:
            raise QC_error("rho and rho a and rho b do not all have the same type")
            
    def quantum_mutual_info(self) -> float:
        if all(isinstance(i, np.ndarray) for i in (self.rho, self.rho_a, self.rho_b)):
            mut_info = self.vn_entropy(self.rho_a) + self.vn_entropy(self.rho_b) - self.vn_entropy(self.rho)
            return mut_info
        else:
            raise QC_error(f"You need to provide rho a, rho b and rho for this computation to work")
    
    def quantum_relative_entropy(self, rho=None) -> float:   #rho is again the first value in S(A||B)  pretty sure this is wrong
        if isinstance(self.rho_a, np.ndarray) and isinstance(self.rho_b, np.ndarray):
            rho_a = np.zeros(len(self.rho_a),dtype=np.complex128)
            rho_b = np.zeros(len(self.rho_b),dtype=np.complex128)
            for i, val in enumerate(self.rho_a):
                rho_a[i] = val
                rho_a[i] += 1e-10 if val == 0 else 0
            for i, val in enumerate(self.rho_b):
                rho_b[i] = val
                rho_b[i] += 1e-10 if val == 0 else 0
            if rho in ["rho_a","a","A"]:
                quant_rel_ent = trace(rho_a*(flatten_matrix(logm(reshape_matrix(rho_a)) - logm(reshape_matrix(rho_b)))))
                return quant_rel_ent
            elif rho in ["rho_b","b","B"]:
                quant_rel_ent = trace(rho_b*(flatten_matrix(logm(reshape_matrix(rho_b)) - logm(reshape_matrix(rho_a)))))
                return quant_rel_ent
        else:
            raise QC_error(f"You need to provide two rhos of the correct type")

    def partial_trace(self, **kwargs) -> np.ndarray:
        trace_out_system = kwargs.get("trace_out", "B")
        trace_out_state_size = int(kwargs.get("state_size", 1))
        rho_length = len(self.rho)
        rho_dim = int(np.sqrt(rho_length))
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        reduced_length = int(reduced_dim**2)
        new_mat = np.zeros(reduced_length,dtype=np.complex128)
        traced_out_dim_range = np.arange(traced_out_dim)
        if isinstance(self.rho, np.ndarray):
            if trace_out_system == "B":
                    for k in range(reduced_dim):
                        for i in range(reduced_dim):
                            new_mat[i+k*reduced_dim] = np.sum(self.rho[traced_out_dim_range+traced_out_dim_range*rho_dim+i*traced_out_dim+k*rho_dim*traced_out_dim])
                    self.rho_a = new_mat
                    return self.rho_a
            elif trace_out_system == "A":
                    for k in range(reduced_dim):
                        for i in range(reduced_dim):
                            new_mat[i+k*reduced_dim] = np.sum(self.rho[reduced_dim*(traced_out_dim_range+traced_out_dim_range*rho_dim)+i+k*rho_dim])
                    self.rho_b = new_mat
                    return self.rho_b
        else:
            QC_error(qc_dat.error_class)

    def __sub__(self, other: "Density") -> "Density":
        new_name: str = f"{self.name} - {other.name}"
        new_mat = np.zeros(self.length,dtype=np.complex128)
        if isinstance(self, Density) and isinstance(other, Density):
            new_mat = self.rho - other.rho
            return Density(name=new_name, info=qc_dat.Density_matrix_info, rho=np.array(new_mat))
        else:
            raise QC_error(qc_dat.error_class)
        
    def __add__(self, other: "Density") -> "Density":
        new_name: str = f"{self.name} + {other.name}"
        new_mat = np.zeros(self.length,dtype=np.complex128)
        if isinstance(self, Density) and isinstance(other, Density):
            new_mat = self.rho + other.rho
            return Density(name=new_name, info=qc_dat.Density_matrix_info, rho=np.array(new_mat))
        else:
            raise QC_error(qc_dat.error_class)


class Measure(Density):
    def __init__(self, **kwargs):
        self.measurement_qubit = kwargs.get("m_qubit", "all")
        self.measure_type: str = kwargs.get("type", "projective")
        self.state = kwargs.get("state", None)
        self.name = kwargs.get("name", f"Measurement of state")
        self.fast = kwargs.get("fast", False)
        if self.fast:
            pass
        else:
            if self.state is not None:
                self.density: Density = kwargs.get("density", Density(state=self.state))
                self.rho: np.ndarray = self.density.rho
            else:
                self.density = kwargs.get("density", None)
                self.rho = self.density.rho if isinstance(self.density, Density) else kwargs.get("rho", None)
            self.measurement = self.measure_state()

    @property
    def length(self) -> int:
        return len(self.rho)
    
    @property
    def dim(self) -> int:
        return int(np.sqrt(self.length))
    
    @property
    def n(self) -> int:
        return int(np.log2(self.dim))

    def __str__(self):
        return f"Measure"
    
    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.list_probs()}[/not bold]"

    def measure_probs(self) -> str:
        if self.measure_type == "projective":
            return self.list_probs()
        
    def topn_measure_probs(self, **kwargs) -> np.ndarray:
        topn = kwargs.get("n", 8)
        return top_probs(self.list_probs(), topn)
    
    def list_probs(self, qubit: int = None, povm: list = None) -> np.ndarray:
        if povm is not None:
            probs = np.array([np.real(np.trace(self.density * E)) for E in povm], dtype=np.float64)
            return probs
        
        if qubit is None:
            if self.fast:
                vector = self.state.vector
                return np.real(np.multiply(vector, np.conj(vector)))
            elif isinstance(self.density, Density):
                if self.rho is None:
                    self.rho = self.density.rho
                return np.array([self.rho[i + i * self.dim].real for i in range(self.dim)], dtype=np.float64)
            else:
                raise QC_error(qc_dat.error_kwargs)
        
    def measure_state(self, qubit: int = None, povm: list = None) -> str:
        PD = self.list_probs(qubit, povm)
        measurement = choices(range(len(PD)), weights=PD)[0]
        if povm is not None:
            return f"Measured POVM outcome: {measurement}"
        
        if qubit is None:
            num_bits = int(np.log2(self.dim))
            result = f"Measured the state: |{bin(measurement)[2:].zfill(num_bits)}>"
            self.state.vector[measurement] = 1
            self.state.vector = self.state.vector / np.linalg.norm(self.state.vector)
            return result

def format_ket_notation(list_probs, **kwargs) -> str:
    list_type = kwargs.get("type", "all")
    num_bits = kwargs.get("num_bits", int(np.ceil(np.log2(len(list_probs)))))
    prec = kwargs.get("precision", 3)
    if list_type == "topn":
        print_out = f""
        for ket_val, prob_val in zip(list_probs[:,0],list_probs[:,1]):
            print_out += (f"State |{bin(ket_val)[2:].zfill(num_bits)}> ({ket_val}) with a prob val of: {prob_val * 100:.{prec}f}%\n")
        return print_out
    elif list_type == "all":
        ket_mat = range(len(list_probs))
        print_out = f""
        for ket_val, prob_val in zip(ket_mat,list_probs):
            print_out += (f"|{bin(ket_val)[2:].zfill(num_bits)}>  {prob_val:.{prec}f}%\n")
        return print_out


class Circuit:
    def __init__(self, **kwargs):
        self.gates = []
        self.state = kwargs.get("state", None)
        if isinstance(self.state, Qubit):
            self.n = int(np.log2(self.state.dim))
        else:
            self.n = kwargs.get("n", None)
            if isinstance(self.n, int):
                new_vector = np.zeros(2**self.n, dtype=np.complex128)
                new_vector[0] = 1
                self.state = Qubit(vector=new_vector)
        self.start_gate: Gate = Gate.Identity(n=self.n)
        console.rule(f"Initialising a Quantum Circuit with {self.n} Qubits", style="circuit_header")
        console.rule("", style="circuit_header")
        self.measurement = None

    def __str__(self):
        return f"{self.final_state}"
    
    def __rich__(self):
        console.rule(f"Running Quantum Circuit with {self.n} Qubits:", style="circuit_header", characters="/")
        print_array(self.final_gate)
        print_array(self.state)
        print_array(f"Final Probability Distribution:")
        print_out = format_ket_notation(self.top_prob_dist, type="topn", num_bits=int(np.ceil(np.log2(self.state.dim))))
        print_array(print_out)
        print_array(f"{self.measure_state}")
        console.rule(f"", style="circuit_header")

    def print_gates(self):
        for gate in reversed(self.gates):
            print_array(gate)

    def add_gate(self, gate: Gate, text=True):
        self.gates.append(gate)
        if text:
            if gate.dim < 9:
                print_array(f"Adding this gate to the circuit:")
                print_array(gate)
            else:
                print_array(f"Adding the {gate.dim} x {gate.dim} gate: {gate.name} to the circuit")

    def add_single_gate(self, gate: Gate, gate_location: int, text=True):
        if self.n:
            if isinstance(gate_location, int):
                upper_id = Gate.Identity(n=gate_location)
                lower_id = Gate.Identity(n=self.n - gate_location - 1)
            else:
                raise QC_error(f"The gate location connot be of {type(gate_location)}, expect type int")
            ndim_gate = upper_id @ gate @ lower_id
            self.gates.append(ndim_gate)
            if text:
                if gate.dim < 9:
                    print_array(f"Adding this gate to the circuit:")
                    print_array(ndim_gate)
                else:
                    print_array(f"Adding the {self.n} x {self.n} gate: {gate.name} to the circuit")

    def compute_final_gate(self, text=True) -> Gate:
        self.final_gate = self.start_gate
        for gate in reversed(self.gates):
            self.final_gate = self.final_gate * gate
        self.final_gate.name = f"Final Gate"
        if text:
            print_array(f"The final Gate is:")
            print_array(self.final_gate)
        
        return self.final_gate
    
    def apply_final_gate(self, text=True) -> Qubit:
        self.state = self.final_gate * self.state
        if text:
            print_array(f"The final state is:")
            print_array(self.state)
        return self.state
    
    def list_probs(self, text=True) -> Measure:
        self.prob_distribution = Measure(state=self.state).list_probs()
        if text:
            print_array(f"The projective probability distribution is:")
            print_array(format_ket_notation(self.prob_distribution))
        return self.prob_distribution
    
    def topn_probabilities(self, text=True, **kwargs) -> Measure:
        topn = kwargs.get("n", 8)
        self.top_prob_dist = top_probs(self.list_probs(text=False), topn)
        if text:
            print_array(f"The top {topn} probabilities are:")
            print_array(format_ket_notation(self.top_prob_dist, type="topn", num_bits=int(np.ceil(np.log2(self.state.dim)))))
        return self.top_prob_dist
    
    def measure_state(self, text=True) -> Measure:
        self.measure_state = Measure(state=self.state).measure_state()
        if text:
            print_array(f"The measured state is:")
            print_array(self.measure_state)
        return self.measure_state

    def run(self):
        if self.gates == []:
            raise QC_error(qc_dat.error_empty_circuit)
        else:
            self.compute_final_gate(text=False)
            self.apply_final_gate(text=False)
            self.list_probs(text=False)
            self.topn_probabilities(text=False)
            self.measure_state(text=False)
        return self.__rich__()

    def return_info(self, attr):
        if not hasattr(self, attr): 
            raise QC_error(f"This parameter {attr} of type {type(attr)} does not exist")
        return getattr(self, attr)  
        
    
    
class Grover:                                               #this is the Grover algorithms own class
    def __init__(self, *args, **kwargs):
        self.fast = kwargs.get("fast", True)
        self.n_cap: int = int(kwargs.get("n_cap",16 if self.fast else 12))         
        self.n = kwargs.get("n", None)
        self.it = kwargs.get("iterations", None)         
        self.oracle_values = []
        self.rand_ov = 0
        self.results = kwargs.get("results", [])
        self.iter_calc = kwargs.get("iter_calc", None)
        self.balanced_param = kwargs.get("balanced_param", 100)
        for arg in args:
            if isinstance(arg, list): 
                self.oracle_values.extend(arg)  
            elif isinstance(arg, int):
                self.rand_ov = arg
        
            

    def __str__(self):
        return f"{self.name}\n{self.results}"
    
    def __rich__(self) -> str:           #creates the correct printout so it shows the prob next to the ket written in ket notation
        print_out = f"[bold]{self.name}[/bold]\n"
        print_out_kets = format_ket_notation(self.results, type="topn", num_bits=int(np.ceil(self.n)), precision = (3 if self.n < 20 else 6))
        print_out = print_out + print_out_kets
        return print_out

    def phase_oracle(self, qub: Qubit, oracle_values: list) -> Qubit:          #the Grover phase oracle
        qub.vector[oracle_values] *= -1 
        return qub
    
    def optimal_iterations(self, n: int) -> tuple[float, int]:
        search_space: int = 2**n
        op_iter: float = (np.pi/4)*np.sqrt((search_space)/len(self.oracle_values)) - 0.5
        return op_iter, search_space
    
    def init_states(self) -> tuple[Qubit, Gate]:
        timer = Timer()
        spec_had_mat = np.array([1,1,1,-1])    #i use this so that all the matrix mults are by an integer value and not a float and then apply the float later
        spec_had = Gate(name="Custom Hadamard for Grovers", info=qc_dat.Hadamard_info, matrix=spec_had_mat)
        qub = Qubit.q0(n=self.n)
        print_array(f"Initialising state {qub.name}")
        had_op = spec_had                      
        print_array(f"Initialising {self.n} x {self.n} Hadamard")
        for i in range(self.n-1):    #creates the qubit and also the tensored hadamard for the given qubit size
            had_op **= spec_had
            print(f"\r{i+2} x {i+2} Hadamard created", end="")    #allows to clear line without writing a custom print function in print_array
        print(f"\r",end="")
        print_array(f"\rHadamard and Quantum State created, time to create was: {timer.elapsed()[0]:.4f}")
        return qub, had_op

    def iterate_alg(self) -> Qubit:
        it = 0
        timer = Timer()
        if self.fast:
            F = FWHT()
            qub = Qubit.q0(n=self.n)
            print_array(f"Running FWHT algorithm:")
            while it < int(self.it):   #this is where the bulk of the computation actually occurs and is where the algorithm is actually applied
                print(f"\rIteration {it + 1}:                                                                  ", end="")
                if it != 0:
                    qub: Qubit = final_state
                print(f"\rIteration {it + 1}: Applying first Hadamard                                          ", end="")
                initialized_qubit = F * qub       #applies a hadamard to every qubit                           STEP 1
                print(f"\rIteration {it + 1}: Applying phase oracle and second Hadamard                        ", end="")
                intermidary_qubit = F * self.phase_oracle(initialized_qubit, self.oracle_values)              #STEP 2   phase flips the given oracle values
                print(f"\rIteration {it + 1}: Flipping the Qubits phase except first Qubit                     ", end="")
                intermidary_qubit.vector *= -1           #inverts all of the phases of the qubit values             STEP 3a
                intermidary_qubit.vector[0] *= -1              #inverts back the first qubits phase                 STEP 3b
                print(f"\rIteration {it + 1}: Applying third and final Hadamard                                ", end="")
                final_state = F * intermidary_qubit        #applies yet another hadamard gate to the qubits    STEP 4
                it += 1                   #adds to the iteration counter
                print(f"\r                                                                                     Time elapsed:{timer.elapsed()[0]:.4f} secs", end="")
            print(f"\r",end="")
            print_array(f"\rFinal state calculated. Time to iterate algorithm: {timer.elapsed()[1]:.4f} secs                                                                        ")
            return final_state
        else:
            qub, had_op = self.init_states()
            print_array(f"Running algorithm:")
            it = 0
            had_norm = 1/np.sqrt(2**self.n)
            while it < int(self.it):   #this is where the bulk of the computation actually occurs and is where the algorithm is actually applied
                print(f"\rIteration {it + 1}:                                                                  ", end="")
                if it != 0:
                    qub: Qubit = final_state
                print(f"\rIteration {it + 1}: Applying first Hadamard                                          ", end="")
                initialized_qubit = had_op * qub       #applies a hadamard to every qubit                           STEP 1
                print(f"\rIteration {it + 1}: Applying phase oracle and second Hadamard                        ", end="")
                intermidary_qubit = had_op * self.phase_oracle(initialized_qubit, self.oracle_values)              #STEP 2   phase flips the given oracle values
                print(f"\rIteration {it + 1}: Flipping the Qubits phase except first Qubit                     ", end="")
                intermidary_qubit.vector *= -1           #inverts all of the phases of the qubit values             STEP 3a
                intermidary_qubit.vector[0] *= -1              #inverts back the first qubits phase                 STEP 3b
                print(f"\rIteration {it + 1}: Applying third and final Hadamard                                ", end="")
                final_state = had_op * intermidary_qubit        #applies yet another hadamard gate to the qubits    STEP 4
                final_state.vector *=  had_norm**3             #applies the normalisation factor here
                it += 1                   #adds to the iteration counter
                print(f"\r                                                                                     Time elapsed:{timer.elapsed()[0]:.4f} secs", end="")
            print(f"\r",end="")
            print_array(f"\rFinal state calculated. Time to iterate algorithm: {timer.elapsed()[1]:.4f} secs                                                                        ")
            return final_state

    def compute_n(self) -> int:
        if isinstance(self.n_cap, int):
            print_array(f"Using up to {self.n_cap} Qubits to run the search")
            max_oracle = max(self.oracle_values)
            n_qubit_min = 1
            while max_oracle > 2**n_qubit_min:             #when picking the qubits, we need enough to allow the search space to be bigger than all the oracle values
                n_qubit_min += 1
            if n_qubit_min > self.n_cap:
                raise QC_error(f"The search space needed for this search is larger than the qubit limit {self.n_cap}.")
            if self.it == None:
                print_array(f"No iteration value given, so will now calculate the optimal iterations")
                n_qubit_range = np.arange(n_qubit_min, self.n_cap + 1, dtype=int)
                if self.iter_calc == None or self.iter_calc == "round":
                    int_val = 0
                    print_array(f"Now computing n for the optimal iteration closest to a whole number")
                    for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                        op_iter = self.optimal_iterations(i)[0]
                        if op_iter >= 1:
                            int_dist: float = op_iter - np.floor(op_iter)  #finds the float value
                            int_dist: float = abs(int_dist-0.5)             #shifts them down so its the distance from any integer
                            print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                            if int_dist > int_val:            #iterates through to find the smallest distance from an integer
                                self.n: int = i
                                int_val: float = int_dist
                    return self.n
                elif self.iter_calc == "floor":
                    int_val = 1
                    print_array(f"Now computing n for the optimal iteration closest to the number below it")
                    for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                        op_iter = self.optimal_iterations(i)[0]
                        if op_iter >= 1:
                            int_dist: float = op_iter - np.floor(op_iter)  #finds the float value
                            print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                            if int_dist < int_val:            #iterates through to find the smallest distance from an integer
                                self.n: int = i
                                int_val: float = int_dist
                    return self.n
                elif self.iter_calc == "balanced":
                    if isinstance(self.balanced_param, int):
                        int_val_floor = 1
                        int_val_round = 0
                        print_array(f"Now computing n for the optimal iteration using a balanced algorithm")
                        for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                            op_iter = self.optimal_iterations(i)[0]
                            if op_iter >= 1:
                                int_dist_floor: float = op_iter - np.floor(op_iter)  #finds the float value
                                int_dist_round: float = 2*abs(int_dist_floor-0.5) 
                                print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                                if int_dist_floor < int_val_floor:            #iterates through to find the smallest distance from an integer
                                    n_floor: int = i
                                    int_val_floor: float = int_dist_floor
                                if int_dist_round > int_val_round:            #iterates through to find the smallest distance from an integer
                                    n_round: int = i
                                    int_val_round: float = int_dist_round
                        if (1-int_val_round) < int_val_floor / self.balanced_param:
                            self.n = n_round
                            self.iter_calc = "round"
                            print_array(f"The optimal iteration is computed through rounding")
                        else:
                            self.n = n_floor
                            self.iter_calc = "floor"
                            print_array(f"The optimal iteration is computed by flooring")
                        return self.n
                    else:
                        return TypeError(f"balanced_param cannot be of type {type(self.balanced_param)}, expected str")
                else:
                    raise TypeError(f"iter_calc cannot be of type {type(self.iter_calc)}, expected str")
            else:
                self.n = n_qubit_min
                print_array(f"Running the given {self.it} iterations with the minimum number of qubits {self.n}")
                return self.n
        else:
            raise QC_error(f"The qubit limit cannot be of {type(self.n_cap)}, expected type int")
    
    def run(self) -> "Grover":     #Grovers algorithm, can input the number of qubits and also a custom amount of iterations
        Grover_timer = Timer()
        if self.rand_ov:
            console.rule(f"Grovers search with random oracle values", style="grover_header")
            self.oracle_values = np.zeros(self.rand_ov)
        else:
            console.rule(f"Grovers search with oracle values: {self.oracle_values}", style="grover_header")
        if self.n == None:               #if the number of qubits required is not given then run:
            self.n = self.compute_n()
            search_space: int = 2**self.n       #computes the final search space for the chosen n
            print_array(f"Using {self.n} Qubits with a search space of {search_space} to get the best accuracy")
        elif isinstance(self.n, int):
            search_space: int = 2**self.n       #computes the search space for the n provided
            print_array(f"Using {self.n} Qubits with a search space of {search_space}")
        else:
            raise QC_error(qc_dat.error_class)

        if self.rand_ov:
            self.oracle_values = []
            for i in range(self.rand_ov):
                self.oracle_values.append(randint(0, 2**self.n - 1))
            self.rand_ov = self.oracle_values

        op_iter = self.optimal_iterations(self.n)[0]
        if self.it == None:     #now picks an iteration value
            if self.iter_calc == "round" or self.iter_calc == None:
                self.it = round(op_iter)
            elif self.iter_calc == "floor":
                self.it = int(np.floor(op_iter))
            else:
                raise QC_error(f"Invalid keyword argument")
            
            if self.it < 1:    #obviously we cant have no iterations so it atleast does 1 iteration
                self.it = 1.0000       #more of a failsafe, will almost certainly be wrong as the percentage diff from the it value to 1 will be large?
                print_array(f"Computing 1 iteration, most likely will not be very accurate")
            else:
                print_array(f"Optimal number of iterations are: {op_iter:.3f}")
                print_array(f"Will run {self.it} iterations")
        elif isinstance(self.it, int):
            print_array(f"Number of iterations to perform are: {self.it}")
        else:
            raise QC_error(f"Iterations cannot be of {type(self.it)}, expected type int")
        final_state = self.iterate_alg()
        print_array(f"Computing Probability Distribution of States")
        final_state = Measure(state=final_state, fast=True)
        print_array(f"Finding the probabilities for the top n Probabilities (n is the number of oracle values)")
        sorted_arr = top_probs(final_state.list_probs(), len(self.oracle_values))         #finds the n top probabilities
        print_array(f"Outputing:")
        output = Grover(final_state.name, n=self.n, results=sorted_arr)         #creates a Grover instance
        if self.rand_ov:
            output.name = f"The States of the Grover Search with random Oracle Values {self.oracle_values}, after {int(self.it)} iterations is: "
        else:
            output.name = f"The States of the Grover Search with Oracle Values {self.oracle_values}, after {int(self.it)} iterations is: "
        print_array(output)                #prints that Grover instance
        console.rule(f"Total Time to run Grover's Algorithm: {Grover_timer.elapsed()[0]:.4f} seconds", style="grover_header")
        print()
        return output              #returns the value


class print_array:    #made to try to make matrices look prettier
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
                for ket_val, prob_val in zip(ket_mat,array.list_probs()):
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


X_Gate = Gate.X_Gate()             #initialises the default gates
Y_Gate = Gate.Y_Gate()
Z_Gate = Gate.Z_Gate()
Identity = Gate.Identity()
Hadamard = Gate.Hadamard()
CNot_flip = Gate.C_Gate(type="inverted" , info=qc_dat.C_Not_matrix,name="CNot_flip")
CNot = Gate.C_Gate(info=qc_dat.C_Not_matrix,name="CNot")
Swap = Gate.Swap()
S_Gate = Gate.P_Gate(theta=np.pi/2)
T_Gate = Gate.P_Gate(theta=np.pi/4)
F = FWHT()
Q = QFT()


oracle_values = [9,4,3,2,5,6,12,15, 16]
oracle_values2 = [1,2,3,4, 664, 77,5, 10, 12,14,16, 333, 334, 335, 400, 401, 41, 42]
oracle_values3 = [4, 5, 30, 41]
oracle_values4 = [500, 5, 4, 7, 8, 9, 99]
oracle_value_test = [1,2,3]
def main():
    Grover(oracle_values2, fast=True, iter_calc="round").run()
    Grover(oracle_values2, fast=True, iter_calc="floor").run()
    Grover(oracle_values2, fast=True, iter_calc="balanced").run()
    print_array(Measure(state=Qubit.q0(n=3)).measure_state())