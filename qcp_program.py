import time
start = time.time()                                      
from random import choices, randint                                            #used for measuring
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, logm
from config import *


class Timer:
    """A basic timer to time the time of functions and also the time of the whole program"""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time

    def elapsed(self):                                   #allows for two timing modes, one for overall and one for multiple increments between calls
        current_time = time.perf_counter()
        interval_time = current_time - self.last_time 
        total_time = current_time - self.start_time 
        self.last_time = current_time
        return interval_time, total_time
    
def prog_end():    #made it to make the code at the end of the program a little neater
    """Runs at the end of the program to call the timer and plot only at the end"""
    if __name__ == "__main__":
        main()                #calls functions in main
    stop = time.time()
    interval: float = stop - start
    console.rule(f"{interval:.3f} seconds elapsed",style="main_header")         #gives final time of the code
    plt.show()                             #makes sure to plot graphs after the function
    
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


def comp_Grover_test(n, **kwargs):
        """Compares the fast Grover and the normal Grover"""
        g_loops = kwargs.get("g_loops", 10)
        warmup_timer = Timer()
        warmup = warmup_timer.elapsed()
        loops = np.arange(2,n+1)
        times_array = np.zeros((n-1, 3))
        for i in loops:
            n=int(i)
            test_timer = Timer()
            for j in range(g_loops): Grover(oracle_value_test, n=n, fast=False).run()         #runs each variant of the function for g_loops times and times each one
            time_slow = test_timer.elapsed()[0]
            test_timer = Timer()
            for j in range(g_loops): Grover(oracle_value_test, n=n, fast=True).run()
            time_fast = test_timer.elapsed()[0]
            times_array[i-2] = np.array([n, time_slow/g_loops, time_fast/g_loops])
        print_array(f"Qubits, s time, f time")
        print_array(times_array)

def time_test(n, fast=True, iterations=None, **kwargs):                 #times a Grovers run over variable n qubits
    """Tests the Grover function and its speed with a number of parameters"""
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
            for j in range(g_loops): Grover(oracle_value_test, n=n, fast=fast, iterations=iterations).run()           #used purely for the "fast" variant
        time = test_timer.elapsed()[0]
        times_array[i-2] = np.array([n, time/g_loops])
    print_array(f"Qubits, time")
    print_array(times_array)


def binary_entropy(prob: float) -> float:
    """Used to calculate the binary entropy of two probabilities"""
    if isinstance(prob, (float, int)):
        if prob ==  0 or prob == 1:
            return 0.0
        else:
            return -prob*np.log2(prob) - (1 - prob)*np.log2(1 - prob)
    raise QC_error(f"Binary value must be a float")
    

class Qubit(StrMixin, LinAlgMixin):                                           #creates the qubit class
    """The class to define and initialise Qubits and Quantum States"""
    array_name = "vector"
    def __init__(self, **kwargs) -> None:
        self.detailed = kwargs.get("detailed", None)
        self.state_type: str = kwargs.get("type", "pure")                   #the default qubit is a single pure qubit |0>
        self.name: str = kwargs.get("name","|Quantum State>")
        self.vector: np.ndarray = np.array(kwargs.get("vector",np.array([1,0])),dtype=np.complex128)
        if not isinstance(self.vector, np.ndarray):
            raise QubitError(f"Attribute self.vector cannot be of type {type(self.vector)}, expected numpy array")
        self.dim: int = len(self.vector) if self.vector.ndim == 1 else len(self.vector[0])                   #used constantly in all calcs so defined it universally
        self.n: int = int(np.log2(self.dim))
        if "vectors" in kwargs:
            if self.state_type == "mixed":              #calls mixed state to handle the kwargs
                    self.build_mixed_state(kwargs)
            elif self.state_type == "seperable":
                self.build_seperable_state(kwargs)
        if self.detailed:
            self.density = Density(state=self)          #this stops it from computing this lengthy calc for every Qubit object created
            self.vne = self.density.vn_entropy()
            if self.state_type == "mixed":
                self.se = self.density.shannon_entropy()           #calcs entropy for classical values, not 100% sure on this yet

    def build_mixed_state(self, kwargs):
        """Takes the kwargs and creates a mixed state, mostly just checks types and extracts numpy arrays if Qubits are given"""
        if isinstance(kwargs.get("vectors")[0], Qubit):
            qubit_vectors = np.array(kwargs.get("vectors",[]))
            vector_list = []
            for i in qubit_vectors:
                vector_list.append(i)
            self.vector = vector_list
        elif isinstance(kwargs.get("vectors")[0], (list, np.ndarray)):
            self.vector = np.array(kwargs.get("vectors",[]),dtype=np.complex128) 
        else:
            raise TypeError(f"Invalid type of type{kwargs.get("vectors")[0]}, expected Qubit class, list or np.ndarray")
        self.weights = kwargs.get("weights", [])
        if len(self.weights) != len(self.vector):
            raise QC_error(f"The state must have one weight value {self.weights}, for each vector {self.vector}")
            
    def build_seperable_state(self, kwargs):
        """Tensors the given parameters of a seperable states together into one vector"""
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
            raise QubitError(f"The Seperable Qubit states cannot be of type {type(qubit_states[0])}, expected Qubit or numpy array")

    @classmethod                 #creates the default qubits, using class methods allows you to define an instance within the class
    def q0(cls, **kwargs):
        """The |0> Qubit"""
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
        """The |1> Qubit"""
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
        """The |+> Qubit"""
        n = 1/np.sqrt(2)
        qp_vector = [n,n]
        return cls(name="|+>", vector=qp_vector)

    @classmethod
    def qm(cls):
        """The |-> Qubit"""
        n = 1/np.sqrt(2)
        qm_vector = [n,-n]
        return cls(name="|->", vector=qm_vector)
    
    @classmethod
    def qpi(cls):
        """The |i> Qubit"""
        n =1/np.sqrt(2)
        qpi_vector = np.array([n+0j,0+n*1j],dtype=np.complex128)
        return cls(name="|i>", vector=qpi_vector)
    
    @classmethod
    def qmi(cls):
        """The |-i> Qubit"""
        n =1/np.sqrt(2)
        qmi_vector = np.array([n+0j,0-n*1j],dtype=np.complex128)
        return cls(name="|-i>", vector=qmi_vector)

    def __ipow__(self: "Qubit", other: "Qubit") -> "Qubit":                 #denoted **=
        if isinstance(self, Qubit) and isinstance(other, Qubit):  
            self = self @ other
            return self
        raise QubitError(f"Error from inputs type {type(self)} and {type(other)}, expected two Qubit class inputs")

    def norm(self: "Qubit") -> None:                 #dunno why this is here ngl, just one of the first functions i tried
        """Normalises a Qubit so that the probabilities sum to 1"""
        normalise = np.sqrt(sum([i*np.conj(i) for i in self.vector]))
        self.vector = self.vector/normalise

    def bloch_plot(self) -> None:  #turn this into a class soon, pretty useless but might be worth for report or presentation
        """A bloch plotter that can plot a single Qubit on the bloch sphere with Matplotlib"""
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
qpi = Qubit.qpi()
qmi = Qubit.qmi()

class Gate(StrMixin, LinAlgMixin, DirectSumMixin):    
    """The class that makes up the unitary matrices that implement the gate functions
        List of functions:
            __matmul__: tensor product
            __mul__: matrix multiplication
            __ipow__: iterative powers
            FWHT: Applies fast walsh hadamard transform to a state
            QFT: Applies the quantum fourier transform to a state
            __add__: allows for addition of two gates
            __iadd__: iterative addition of gates
            __and__: used as a placeholder for direct sum
            __iand__: iterative direct sum"""
    array_name = "matrix"
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", None)
        self.info = kwargs.get("info", None)
        self.matrix = np.array(kwargs.get("matrix", None),dtype=np.complex128)
        if self.matrix is None:
            raise GateError(f"Gates can only be initialised if they are provided with a matrix")
        self.length: int = len(self.matrix)
        self.dim: int = int(np.sqrt(self.length))
        self.n: int =  0 if self.dim == 0 else int(np.log2(self.dim))

    @classmethod                #again creates some of the default gates, for ease of use and neatness
    def X_Gate(cls):
        """The X Gate, which can flip the Qubits in the X or computational basis"""
        X_matrix = [0,1,1,0]
        return cls(name="X Gate", matrix=X_matrix)
    
    @classmethod
    def Y_Gate(cls):
        """The Y Gate, which can flip the Qubits in the Y basis"""
        Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]
        return cls(name="Y Gate", matrix=Y_matrix)

    @classmethod
    def Z_Gate(cls):
        """The Z Gate, which can flip the Qubits in the Z basis or |+>, |-> basis"""
        Z_matrix = [1,0,0,-1]
        return cls(name="Z Gate", matrix=Z_matrix)

    @classmethod
    def Identity(cls, **kwargs):    
        """The identity matrix, used mostly to represent empty wires in the circuit
        Args:
            n: int: creates an Identity matrix for n qubits, default is 1 Qubit
        Returns:
            Gate, the identity gate with either custom qubits or for a single Qubit"""
        n = kwargs.get("n", 1)
        if isinstance(n, int):
            if n == 0:
                new_mat = np.array([1], dtype=np.complex128)     #better to set up like this than just not make it for some functions that require tensor products
                return cls(name="Identity Gate", matrix=new_mat, dim=0)
            dim = int(2**n)
            new_mat = np.zeros(dim**2, dtype=np.complex128)
            for i in range(dim):
                new_mat[i+ dim * i] += 1
            return cls(name="Identity Gate", matrix=new_mat)
        else: 
            Id_matrix = [1,0,0,1]
            return cls(name="Identity Gate", matrix=Id_matrix)
    
    @classmethod          
    def Hadamard(cls):
        """THe Hadamard Gate, commonly used to rotate between the computational or X basis and the |+>, |-> or Z basis"""
        n = 1/np.sqrt(2)
        H_matrix = [n,n,n,-n]
        return cls(name="Hadamard", matrix=H_matrix)

    @classmethod                                        #allows for the making of any phase gate of 2 dimensions
    def P_Gate(cls, theta, **kwargs):
        """The phase Gate used to add a local phase to a Qubit"""
        name = kwargs.get("name", f"Phase Gate with a phase {theta:.3f}")    #allows custom naming to creats S and T named gates
        P_matrix = [1,0,0,np.exp(1j*theta)]
        return cls(name=name, matrix=P_matrix)

    @classmethod                                #allows for any unitary gates with three given variables
    def U_Gate(cls, a, b, c):
        """The Unitary Gate, which can approximate nearly any unitary 2 x 2 Gate"""
        U_matrix = [np.cos(a/2),
                    -np.exp(np.complex128(0-1j)*c)*np.sin(a/2),
                    np.exp(np.complex128(0+1j)*b)*np.sin(a/2),                  #not hugely used, just thought it was cool to implement
                    np.exp(np.complex128(0+1j)*(b+c))*np.cos(a/2)]
        return cls(name=f"Unitary Gate with values (a:{a:.3f}, b:{b:.3f}, c:{c:.3f})", matrix=U_matrix)

    @classmethod                             #creates any specific control gate
    def C_Gate(cls, **kwargs):
        """The Control Gate, commonly seen in the form of the CNOT Gate, used to entangle Qubits
        Args:
            type: str: can either give "standard" type or "inverted" type
            gate: Gate: selects the gate action, eg X for CNOT, defaults to X_Gate
        Returns:
            Gate: The specified Control Gate"""
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
        """The Swap Gate, used to flip two Qubits in a circuit"""
        n_is_2 = [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1]
        n: int = kwargs.get("n", 2)
        if n == 2:
            return cls(name=f"2 Qubit Swap gate", matrix=n_is_2)


    def __ipow__(self, other: "Gate") -> "Gate":    #denoted **=
        if isinstance(self, Gate):  
            self = self @ other
            return self
        raise GateError(f"Error from inputs type {type(self)} and {type(other)}, expected two Gate class inputs")
        
    def FWHT(self, other: Qubit) -> Qubit:
        """The Fast Walsh Hadamard Transform, used heavily in Grover's to apply the tensored Hadamard"""
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
                vec[indices + half_step] = (a - b) * sqrt2_inv                        #normalisation has been taken out giving a slight speed up in performance
            return other
        raise TypeError(f"This can't act on this type, only on Qubits")

    @staticmethod
    def fractional_binary(qub: Qubit,m: int) -> float:             #for shors
        """The fractional binary calculator used solely in the Quantum Fourier Transform"""
        bin_name_check = qub.name[1:-2]
        if all(b in "01" for b in bin_name_check):
            num_bits = int(np.ceil(np.log2(qub.dim)))
            x_vals = qub.name[1:1+num_bits]        #creates the value from the name as the default names for Qubits have their binary ket rep, is a little risky tho
            frac_bin = ("0." + x_vals)             #as custom names or error in names will result in an error
            val = 0
            if m <= num_bits:
                for i in range(m):
                    val += float(frac_bin[i+2])*2**-(i+1)
                return val
        raise QC_error(f"The fractional binary reads the binary values in the name of the Qubit, the name of this Qubit is not made of binary values, Qubit name is {qub.name}")

    def QFT(self, other: Qubit) -> Qubit:          #also for shors although used in other algorithms
        """The Quantum Fourier Transform, for now can only act on a Qubit and is not a matrix"""
        old_name = other.name
        n = int(np.ceil(np.log2(other.dim)))
        frac_init = Gate.fractional_binary(other,1)
        four_qub_init = Qubit(vector=np.array([1,np.exp(2*1j*np.pi*frac_init)]))
        four_qub_init.norm()
        four_qub_sum = four_qub_init
        for j in range(n-1):
            frac = Gate.fractional_binary(other,j+2)
            four_qub = Qubit(vector=np.array([1,np.exp(2*1j*np.pi*frac)]))
            four_qub.norm()
            four_qub_sum **= four_qub
        four_qub_sum.name = f"QFT of {old_name}"
        return four_qub_sum

    @staticmethod
    def mul_flat(first, second) -> np.ndarray:
        """The matrix multiplier for flat arrays, used heavily in the dunder method __mul__"""
        if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
            new_mat = np.zeros(len(first),dtype=np.complex128)
            dim = int(np.sqrt(len(first)))
            dim_range = np.arange(dim)
            for i in range(dim):
                for k in range(dim):
                    new_mat[k+(i * dim)] = np.sum(first[dim_range+(i * dim)]*second[k+(dim_range* dim)])
            return new_mat
        raise GateError(f"The inputted parameters have the wrong type of type {type(first)} and {type(second)}, expected two numpy arrays")

    def __mul__(self, other):       #matrix multiplication
        """The matrix multiplier, allowing multiple types of Gates and also applying Gates to Qubits"""
        if isinstance(self, FWHT) and isinstance(other, Qubit):
            return self.FWHT(other)
        elif isinstance(self, QFT) and isinstance(other, Qubit):
            return self.QFT(other)
        elif isinstance(self, Gate):
            if isinstance(other, np.ndarray):
                if isinstance(self, Density):
                    new_rho = Gate.mul_flat(self.rho, other)
                    return Density(rho=new_rho, name=self.name)
                else:
                    new_mat = Gate.mul_flat(self.matrix, other)
                    return Gate(matrix=new_mat, name=self.name)
            if self.dim == other.dim:
                new_name = f"{self.name} * {other.name}"
                if isinstance(other, Qubit):  #splits up based on type as this isnt two n x n but rather n x n and n matrix
                        new_mat = np.zeros(self.dim,dtype=np.complex128)
                        for i in range(self.dim):
                            row = self.matrix[i * self.dim:(i + 1) * self.dim]
                            new_mat[i] = np.sum(row[:] * other.vector[:])
                        return Qubit(vector=new_mat, name=new_name)
                elif isinstance(other, Gate):    #however probs completely better way to do this so might scrap at some point
                        if isinstance(other, Density):
                            new_rho = Gate.mul_flat(self.matrix, other.rho)
                            new_info = "This is the density matrix of: "f"{self.name}"" and "f"{other.name}"
                            return Density(name=new_name, info=new_info, rho=new_rho)
                        else:
                            new_mat = Gate.mul_flat(self.matrix, other.matrix)
                            new_info: str = "This is a matrix multiplication of gates: "f"{self.name}"" and "f"{other.name}"
                            return Gate(name=new_name, info=new_info, matrix=new_mat)
                raise TypeError(f"The second matrix is of type {type(other)} and isn't compatible")
            raise GateError(f"Both the gates must be of the same dimension to perform matrix multiplication. The gates have dim {self.dim} and {other.dim}")
        elif isinstance(self, np.ndarray):
            if isinstance(other, Qubit):  #splits up based on type as this isnt two n x n but rather n x n and n matrix
                new_mat = np.zeros(self.dim,dtype=np.complex128)
                mat_len = len(self)
                mat_dim = int(np.sqrt(mat_len))
                if mat_dim == other.dim:
                    for i in range(mat_dim):
                        row = self[i * mat_dim:(i + 1) * mat_dim]
                        new_mat[i] = np.sum(row[:] * other.vector[:])
                    return Qubit(vector=new_mat, name=self.name)
                raise GateError(f"Both the gates must be of the same dimension to perform matrix multiplication. The gates have dim {mat_dim} and {other.dim}")
            elif isinstance(other, np.ndarray):
                new_mat = Gate.mul_flat(self, other)
                return new_mat
            elif isinstance(other, Gate):
                new_mat = Gate.mul_flat(self, other.matrix)
                return Gate(matrix=new_mat)
        raise GateError(f"Matrix multiplication cannot occur with classes {type(self)} and {type(other)}")

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
        self.name = f"Quantum Fourier Transform"
        self.info = f"An efficient way to compute the Quantum Fourier transform over all qubits in a state"
        self.matrix = np.zeros(4)
        self.length: int = len(self.matrix)
        self.dim: int = int(np.sqrt(self.length))
        self.n: int =  0 if self.dim == 0 else int(np.log2(self.dim))


class Density(Gate):       #makes a matrix of the probabilities, useful for entangled states
    """The class that computes density matrices of Quantum states and can provide probability information for the state
        List of functions:
        generic density: Computes the density matrix of a pure state
        mixed density: Computes the density matrix of a mixed state
        fidelity: finds the fidelity between two states
        trace distance: finds the trace distance between two states
        vn entorpy: finds the von neumann entropy of a single state
        shannon entropy: finds the shannon entropy of a single state
        quantum conditional entropy: finds the qce between two states
        quantum mutual information: finds the qmi between two states
        quantum relaitve entropy: finds the qre between two states
        partial trace: can trace out a subsystem from a wider system"""
    
    array_name = "rho"
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
            raise DensityError(f"The inputted state must be a Qubit class, not of type {type(self.state)}")
        self.state_a = kwargs.get("state_a", None)
        self.state_b = kwargs.get("state_b", None)
        self.rho_a = kwargs.get("rho_a", None if self.state_a is None else self.construct_density_matrix(self.state_a))
        self.rho_b = kwargs.get("rho_b", None if self.state_b is None else self.construct_density_matrix(self.state_b))
        self.rho = kwargs.get("rho", None if self.state is None else self.construct_density_matrix(self.state))
        if self.rho is not None:
            self.length = len(self.rho) if self.state is None else self.state.dim**2
            self.dim = int(np.sqrt(self.length))
            self.n = int(np.log2(self.dim))
        
    def construct_density_matrix(self, calc_state=None) -> np.ndarray:
        """Assigns which density constructor to use for the given Qubit type"""
        if isinstance(calc_state, Qubit):
            if calc_state.state_type in ["pure", "seperable", "entangled"]:
                return self.generic_density(calc_state)
            elif calc_state.state_type == "mixed":
                return self.mixed_density(calc_state)

    def generic_density(self, calc_state: Qubit, **kwargs) -> np.ndarray:       #taken from the old density matrix function
        """Computes the density matrix for any pure combination of states"""
        state_vector = calc_state.vector
        calc_state_dim = len(state_vector)
        rho = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
        qubit_conj = np.conj(state_vector)
        for i in range(calc_state_dim):
            for j in range(calc_state_dim):
                rho[j+(i * calc_state_dim)] += qubit_conj[i]*state_vector[j]
        if abs(1 -trace(rho)) < 1e-5:
            return rho
        raise QC_error(f"The trace of a density matrix must be 1, calculated trace is {trace(rho)}")
            
    def mixed_density(self, calc_state: Qubit, **kwargs) -> np.ndarray:
        """Computes the density matrix for a mixed Qubit state"""
        state_vector = calc_state.vector
        if isinstance(state_vector[0], np.ndarray):
            state_vector: np.ndarray = calc_state.vector
            calc_state_dim = len(state_vector[0])         
        elif isinstance(state_vector[0], Qubit):
            state_vector = np.zeros((len(calc_state.vector),calc_state.dim), dtype=np.complex128)
            for i in range(len(calc_state.vector)):
                state_vector[i] = calc_state.vector[i].vector          #takes the vectors out of the Qubit object to compute
            calc_state_dim = len(state_vector[0])
        qubit_conj: np.ndarray = np.conj(state_vector)
        rho = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
        rho_sub = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
        for k in range(len(state_vector)):
            for i in range(calc_state_dim):        #creates the density matrix for each state vector
                for j in range(calc_state_dim):
                    rho_sub[j+(i * calc_state_dim)] += qubit_conj[k][i]*state_vector[k][j]
            rho += calc_state.weights[k]*rho_sub
            rho_sub = np.zeros(calc_state_dim*calc_state_dim,dtype=np.complex128)
        rho_trace: float = trace(rho)
        if abs(1 - rho_trace) < 1e-5:     #checks its computed properly
            return rho
        raise QC_error(f"The trace of a density matrix must be 1, calculated trace is {trace(rho)}")

    def fidelity(self, rho_1: np.ndarray=None, rho_2: np.ndarray=None) -> float:
        """Computes the fidelity between two states
        Args:
            self: The Density instance
            rho_1: Defaults to self.rho_a, but can take any given rho
            rho_2: Defaults to self.rho_b, but can take any given rho
        Returns:
            self.fidelity_ab: if self.rho_a and self.rho_b are the default
            fidelity: if using custom rho_1 and rho_2"""
        if self.rho_a is not None and rho_1 is None:
            rho_1 = self.rho_a
        if self.rho_b is not None and rho_2 is None:
            rho_2 = self.rho_b
            if isinstance(rho_1, np.ndarray) and isinstance(rho_2, np.ndarray):
                rho_1 = reshape_matrix(rho_1)
                sqrt_rho1: np.ndarray = sqrtm(rho_1)           #has to be 2D to use this function
                flat_sqrt_rho1 = flatten_matrix(sqrt_rho1)     #flattens back down for matrix multiplication
                product =  flat_sqrt_rho1 * rho_2 * flat_sqrt_rho1
                sqrt_product = sqrtm(reshape_matrix(product))             #reshapse and sqrts in the same line
                flat_sqrt_product = flatten_matrix(sqrt_product)           #flattens to take trace of, could be more efficient to use np.trace but i like avoiding external functions where possible
                mat_trace = trace(flat_sqrt_product)
                mat_trace_conj = np.conj(mat_trace)
                fidelity = (mat_trace*mat_trace_conj).real
                if rho_1 is None and rho_2 is None:
                    self.fidelity_ab = fidelity
                    return self.fidelity_ab
                return fidelity

    def trace_distance(self, rho_1: np.ndarray=None, rho_2: np.ndarray=None) -> float:
        """Computes the trace distance of two states
        Args:
            self: The Density instance
            rho_1: Defaults to self.rho_a, but can take any given rho
            rho_2: Defaults to self.rho_b, but can take any given rho
        Returns:
            self.trace_dist: if self.rho_a and self.rho_b are the default
            trace_dist: if using custom rho_1 and rho_2"""
        if self.rho_a is not None and rho_1 is None:
            rho_1 = self.rho_a
        if self.rho_b is not None and rho_2 is None:
            rho_2 = self.rho_b
        diff_mat = rho_1 - rho_2
        rho_dim = int(np.sqrt(len(rho_1)))
        dim_range = np.arange(rho_dim)
        trace_dist = np.sum(0.5 * np.abs(diff_mat[dim_range + rho_dim * dim_range]))
        if rho_1 is None and rho_2 is None:
            self.trace_dist = trace_dist
            return self.trace_dist
        return trace_dist
        
    def vn_entropy(self, rho: np.ndarray=None) -> float:
        """Computes the Von Neumann entropy of a state
        Args:
            self: The Density instance
            rho: Defaults to self.rho if no rho given
                 Can compute vne for any given rho
        Returns:
            float: The Von Neumann entropy
        """
        if self.rho is not None and rho is None:
            rho = self.rho
        if isinstance(rho, np.ndarray):
            reshaped_rho = reshape_matrix(rho)
            eigenvalues, eigenvectors = np.linalg.eig(reshaped_rho)
            entropy = 0
            for ev in eigenvalues:
                if ev > 0:    #prevents ev=0 which would create an infinity from the log2
                    entropy -= ev * np.log2(ev)
            if entropy < 1e-10:         #rounds the value if very very small
                entropy = 0.0
            return entropy
        raise DensityError(f"No rho matrix provided")
        
    def shannon_entropy(self, state: Qubit=None) -> float:
        """computes the shannon entropy of a mixed state
        Args:
            self: The density instance
            state: Defaults to self.state if no state given
                   Can compute se for any given state
        Returns:
                float: The Shannon entropy"""
        if self.state is not None and state is None:
            if isinstance(self.state, Qubit):
                state = self.state
            else:
                raise DensityError(f"self.state is of type {type(self.state)}, expected Qubit class")
        if isinstance(state, Qubit) and state.state_type == "mixed":
            entropy = 0
            for weights in state.weights:
                if weights > 0:    #again to stop infinities
                    entropy -= weights * np.log2(weights)
            if entropy < 1e-10:
                entropy = 0.0
            return entropy
        raise DensityError(f"No mixed Quantum state of type Qubit provided")
        
    def quantum_conditional_entropy(self, rho_a: np.ndarray=None, rho_b: np.ndarray=None) -> float:    #rho is the one that goes first in S(A|B)
        """Computes the quantum conditional entropy of two states
        Args:
            self: The density instance
            rho_a: Defaults to self.rho_a, but can take any given rho
            rho_b: Defaults to self.rho_b, but can take any given rho
        Returns:
            self.cond_ent: if self.rho_a and self.rho_b are the default
            cond_ent: if using custom rho_a and rho_b"""
        if isinstance(self.rho_a, np.ndarray) and isinstance(self.rho_b, np.ndarray):
            if rho_a is None:
                rho_1 = self.rho_a
            if rho_b is None:
                rho_2 = self.rho_b
        else:
            raise DensityError(f"Incorrect type {type(self.rho_a)} and type {type(self.rho_b)}, expeted both numpy arrays")
        cond_ent = self.vn_entropy(rho_1) - self.vn_entropy(rho_2)
        if rho_a is None and rho_b is None:
            self.cond_ent = cond_ent
            return self.cond_ent
        return cond_ent
            
    def quantum_mutual_info(self) -> float:                   #S(A:B)
        """Computes the quantum mutual information of a system
        Args:
            self: The density instance
                  Takes the internal self.rho, self.rho_a and self.rho_b as inputs
                  self.rho is the overall system while rho_a and rho_b are the traced out components of the system
        Returns:
            float: returns the quantum mutual information of the three rho matrices"""
        if all(isinstance(i, np.ndarray) for i in (self.rho, self.rho_a, self.rho_b)):   #compact way to check all three variables
            mut_info = self.vn_entropy(self.rho_a) + self.vn_entropy(self.rho_b) - self.vn_entropy(self.rho)
            return mut_info
        raise DensityError(f"You need to provide rho a, rho b and rho for this computation to work")
    
    def quantum_relative_entropy(self, rho_a:np.ndarray=None, rho_b:np.ndarray=None) -> float:   #rho is again the first value in S(A||B)
        """Computes the quantum relative entropy of two Quantum states
        Args:
            self: The density instance
            rho_a: Defaults to self.rho_a, but can take any given rho
            rho_b: Defaults to self.rho_b, but can take any given rho
        Returns:
            float: returns the quantum relative entropy"""
        if isinstance(self.rho_a, np.ndarray) and isinstance(self.rho_b, np.ndarray):
            if rho_a is None:
                rho_a = self.rho_a
            if rho_b is None:
                rho_b = self.rho_b
        if isinstance(rho_a, np.ndarray) and isinstance(rho_b, np.ndarray):
            rho_1 = np.zeros(len(rho_a),dtype=np.complex128)
            rho_2 = np.zeros(len(rho_b),dtype=np.complex128)
            for i, val in enumerate(rho_a):
                rho_1[i] = val
                rho_2[i] += 1e-10 if val == 0 else 0          #regularises it for when the log is taken
            for i, val in enumerate(rho_b):
                rho_1[i] = val
                rho_2[i] += 1e-10 if val == 0 else 0
            quant_rel_ent = trace(rho_1*(flatten_matrix(logm(reshape_matrix(rho_1)) - logm(reshape_matrix(rho_2)))))
            return quant_rel_ent
        raise DensityError(f"Incorrect type {type(self.rho_a)} and type {type(self.rho_b)}, expected both numpy arrays")

    def partial_trace(self, **kwargs) -> np.ndarray:
        """Computes the partial trace of a state, can apply a trace from either 'side' and can trace out an arbitrary amount of qubits
        Args:
            self: The density instance
            **kwargs
            trace_out_system:str : Chooses between A and B which to trace out, defaults to B
            state_size:int : Chooses the number of Qubits in the trace out state, defaults to 1 Qubit
        Returns:self.rho_a if trace_out_system = B
                self.rho_b if trace_out_system = A"""
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
                        for i in range(reduced_dim):           #the shapes of tracing A and B look quite different but follow a diagonalesc pattern
                            new_mat[i+k*reduced_dim] = np.sum(self.rho[traced_out_dim_range+traced_out_dim_range*rho_dim+i*traced_out_dim+k*rho_dim*traced_out_dim])
                    self.rho_a = new_mat
                    return self.rho_a
            elif trace_out_system == "A":
                    for k in range(reduced_dim):
                        for i in range(reduced_dim):
                            new_mat[i+k*reduced_dim] = np.sum(self.rho[reduced_dim*(traced_out_dim_range+traced_out_dim_range*rho_dim)+i+k*rho_dim])
                    self.rho_b = new_mat
                    return self.rho_b
        raise DensityError(f"self.rho cannot be of type {type(self.rho)}, expected numpy array")


class Measure(StrMixin, LinearMixin):
    """The class in which all measurements and probabilities are computed"""
    array_name = "probs"
    def __init__(self, **kwargs):
        self.measurement_qubit = kwargs.get("m_qubit", "all")
        self.measure_type: str = kwargs.get("type", "projective")
        self.state = kwargs.get("state", None)
        self.name = kwargs.get("name", f"Measurement of state")
        self.fast = kwargs.get("fast", False)
        if not self.fast:
            if self.state is not None:
                self.density: Density = kwargs.get("density", Density(state=self.state))
                self.rho: np.ndarray = self.density.rho
                if self.rho is not None:
                    self.length = len(self.rho)
                    self.dim = int(np.sqrt(self.length))
                    self.n = int(np.log2(self.dim))
            else:
                self.density = kwargs.get("density", None)
                self.rho = self.density.rho if isinstance(self.density, Density) else kwargs.get("rho", None)
                if self.rho is not None:
                    self.length = len(self.rho)
                    self.dim = int(np.sqrt(self.length))
                    self.n = int(np.log2(self.dim))
        self.probs = self.list_probs()
        self.pm_state = None
        self.measurement = None

    def topn_measure_probs(self, qubit: int=None, povm: np.ndarray=None, **kwargs) -> np.ndarray:
        """Gives the top n probabilities"""
        topn = kwargs.get("n", 8)
        return top_probs(self.list_probs(qubit, povm), topn)
    
    def list_probs(self, qubit: int=None, povm: np.ndarray=None) -> np.ndarray:
        """Gives the prob lists of a Quantum state, can measure non projectively for the whole state and projectively for single Qubits"""
        if povm is not None:
            self.probs = np.array([np.real(trace(P * self.density.rho)) for P in povm], dtype=np.float64)
            return self.probs
        if qubit is None:
            if self.fast:
                vector = self.state.vector
                return np.real(np.multiply(vector, np.conj(vector)))
            elif isinstance(self.density, Density):
                if self.rho is None:
                    self.rho = self.density.rho
                self.probs = np.array([self.rho[i + i * self.density.dim].real for i in range(self.density.dim)], dtype=np.float64)
                return self.probs
            raise MeasurementError(f"Must either be running in fast, or self.density is of the wrong type {type(self.density)}, expected Density class")
        if qubit is not None:
            if qubit > self.n - 1:
                raise QC_error(f"The chosen qubit {qubit}, must be no more than the number of qubits in the circuit {self.n}")
            trace_density = self.density
            if qubit == 0:
                A_rho = np.array([1+1j])
                B_rho = trace_density.partial_trace(trace_out="A", state_size = 1)
                measure_rho = trace_density.partial_trace(trace_out="B", state_size = self.n - 1)
            elif qubit == self.n - 1:
                A_rho = trace_density.partial_trace(trace_out="B", state_size = 1)
                B_rho = np.array([1+1j])
                measure_rho = trace_density.partial_trace(trace_out="A", state_size = self.n - 1)
            elif isinstance(qubit, int):
                A_rho = trace_density.partial_trace(trace_out="B", state_size = self.n - qubit)
                B_rho = trace_density.partial_trace(trace_out="A", state_size = qubit + 1)
                measure_den = Density(rho=A_rho)
                measure_rho = measure_den.partial_trace(trace_out="A", state_size = measure_den.n - 1)
            else:
                raise MeasurementError(f"Inputted qubit cannot be of type {type(qubit)}, expected int") 
            
            measure_den = Density(rho=measure_rho)
            if povm is not None:
                self.probs = np.array([np.real(trace(P * measure_den.rho)) for P in povm], dtype=np.float64)
                return self.probs, measure_rho, A_rho, B_rho
            if povm is None:
                self.probs = np.array([measure_den.rho[i + i * measure_den.dim].real for i in range(measure_den.dim)], dtype=np.float64)
                return self.probs, measure_rho, A_rho, B_rho
            
    def measure_state(self, qubit: int = None, povm: np.ndarray = None, text: bool = False) -> str:
        """Measures the state and also computes the collapsed state. Can measure non projectively for all Qubits and projectively for single Qubits"""
        if qubit is not None:
            probs, measure_rho, A_rho, B_rho = self.list_probs(qubit, povm)
        elif qubit is None:
            probs = self.list_probs(qubit, povm)
        self.measurement = choices(range(len(probs)), weights=probs)[0]
        if povm is not None:
            if text:
                print_array(f"Measured POVM outcome: {povm[self.measurement]}")
            return self.measurement
        
        if qubit is None:
            num_bits = int(np.log2(self.state.dim))
            if text:
                print_array(f"Measured the state: |{bin(self.measurement)[2:].zfill(num_bits)}>")
            self.state.vector[:] = 0
            self.state.vector[self.measurement] = 1
            self.state.vector = self.state.vector / np.linalg.norm(self.state.vector)
            return self.measurement, self.state
        elif isinstance(qubit, int):
            if self.measurement == 0:
                measure_rho = np.array([1,0,0,0])
            elif self.measurement == 1:
                measure_rho = np.array([0,0,0,1])
            num_bits = int(np.log2(1))
            if text:
                print_array(f"Measured the {qubit} qubit in state |{bin(self.measurement)[2:].zfill(num_bits)}>")
            post_measurement_den = Density(rho=A_rho) @ Density(rho=measure_rho) @ Density(rho=B_rho)
            pm_state = Qubit(vector=diagonal(post_measurement_den.rho))
            pm_state.norm()
            return self.measurement, pm_state
        else:
            MeasurementError(f"Inputted qubit cannot be of type {type(qubit)}, expected int") 


X_Gate = Gate.X_Gate()             #initialises the default gates
Y_Gate = Gate.Y_Gate()
Z_Gate = Gate.Z_Gate()
Identity = Gate.Identity()
Hadamard = Gate.Hadamard()
CNot_flip = Gate.C_Gate(type="inverted", name="CNot_flip")
CNot = Gate.C_Gate(type="standard", name="CNot")
Swap = Gate.Swap()
S_Gate = Gate.P_Gate(theta=np.pi/2, name="S Gate")
T_Gate = Gate.P_Gate(theta=np.pi/4, name="T Gate")
F = FWHT()
Q = QFT()


class Circuit(BaseMixin):
    """The compiler to run an actual circuit with"""
    array_name = "state"
    def __init__(self, **kwargs):
        self.collapsed = False        #prevents additional gates after collapsing the entire state
        self.measured_states = []
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
        if self.state.n != self.n:
            raise QuantumCircuitError(f"The initial Quantum state of qubit size {self.state.n} must have the same qubits as the size of the circuit of size {self.n}")
        self.start_gate: Gate = Gate.Identity(n=self.n)
        console.rule(f"Initialising a Quantum Circuit with {self.n} Qubits", style="circuit_header")
        console.rule("", style="circuit_header")
        self.density = Density(state=self.state)
        self.measurement = None
        self.final_gate = None
        self.noisy = kwargs.get("noisy", False)
        if self.noisy:     #checks that it has all of the required key word arguments if put in noisy mode
            self.Q_channel = kwargs.get("Q_channel", None)
            self.prob = kwargs.get("prob", None)
            if self.Q_channel == None:
                raise QuantumCircuitError(f"If the Quantum circuit is noisy, a Quantum channel must be given")
            if self.prob == None:
                raise QuantumCircuitError(f"If the Quantum circuit is noisy, an error probability must be given")
            print_array(f"Simulating {self.Q_channel} noise on all gates with an error probability of {self.prob}")
        
    def __str__(self):
        return self.__rich__()
    
    def __rich__(self):
        console.rule(f"Running Quantum Circuit with {self.n} Qubits:", style="circuit_header", characters="/")
        print_array(self.final_gate)
        print_array(f"Final Probability Distribution:")
        print_out = format_ket_notation(self.top_prob_dist, type="topn", num_bits=int(np.ceil(np.log2(self.state.dim))))
        print_array(print_out)
        print_array(f"{self.state}")
        console.rule(f"", style="circuit_header")

    def print_gates(self) -> None:
        """Just used to print specific gates, mostly for debugging purposes"""
        for gate in reversed(self.gates):
            print_array(gate)

    def add_quantum_channel(self, Q_channel: str, prob: float, text: bool=True) -> Qubit:

        if self.collapsed == True:
            raise MeasurementError(f"Noise cannot be applied to a collapsed state")
        K0: Gate = Gate.Identity(n=self.n)
        if Q_channel == "P flip":                  #creates the Kraus operators for the phase flip
            K1 = Gate.Z_Gate()
            Z_Gate = K1
            for i in range(self.n - 1):
                K1 = K1 @ Z_Gate
        elif Q_channel == "B flip":                #creates the Kraus opertors for the bit flip
            K1 = Gate.X_Gate()
            X_Gate = K1
            for i in range(self.n - 1):
                K1 = K1 @ X_Gate
        elif Q_channel == "B P flip":              #creates the Kraus opertors for the Y flip or phase and bit flip
            K1 = Gate.Y_Gate()
            Y_Gate = K1
            for i in range(self.n - 1):
                K1 = K1 @ Y_Gate
        else:
            raise QuantumCircuitError(f"{Q_channel} is not a valid Quantum channel")
        K0.matrix = K0.matrix * np.sqrt(1 - prob)**self.n         #normalisation
        K1.matrix = K1.matrix * np.sqrt(prob)**self.n
        kraus_operators = [K0,K1]
        epsilon_vector = np.zeros(self.state.dim, dtype=np.complex128)
        epsilon = Qubit(vector=epsilon_vector)
        for k in kraus_operators:
            k_conj = k
            k_conj.matrix = reshape_matrix(k_conj.matrix)
            k_conj.matrix = np.conj(k_conj.matrix.T)
            k_conj.matrix = flatten_matrix(k_conj.matrix)
            k_applied = k_conj * self.state           #applies them all to the state
            epsilon += k_applied
            if np.all(epsilon.vector == 0):
                epsilon.vector += 1e-5
        self.state.vector: np.ndarray = epsilon.vector
        self.state.norm()                  #not the most elegent way to do it but only way ive found that works
        self.state.name = new_name
        if not isinstance(self.state.vector, np.ndarray):
            raise QC_error(f"self.state.vector cannot be of type {type(self.state.vector)}, expected numpy array")
        if text:
            print_array(f"Applying the noisy Quantum channel with these Kraus operators:\n")
            print_array(K0)
            print_array(K1)
        return self.state
  
    def add_gate(self, gate: Gate, text: bool=True) -> None:
        """Used to add a combined n x n gates to the gate array to then combine into one unitary gate
            Args:
                self: The Quantum circuit
                gate: Gate: The n x n gate that you would like to add
                text: bool: if True, prints out what its doing
            Returns:
                Nothing: purely adds the gate to self.gates"""
        if self.collapsed == True:
            raise MeasurementError(f"The state is now fully collapsed and no more gates can be applied to it")
        self.gates.append(gate)         #adds the given gate to the array
        if text:
            if gate.dim < 9:
                print_array(f"Adding this gate to the circuit:")
                print_array(gate)
            else:
                print_array(f"Adding the {gate.dim} x {gate.dim} gate: {gate.name} to the circuit")

    def add_single_gate(self, gate: Gate, gate_location: int, text: bool=True):
        """Adds a gate to a single qubit rather than inputting the entire combined gate
            Args:
                self: The Quantum circuit
                gate: Gate: The gate that you would like to add
                gate_location: int: the position in which to add the gate, indexing starts at 0
                text: bool: if True, prints out what its doing
            Returns:
                Nothing: purely adds the gate to self.gates after tensoring with identity gates"""
        if self.collapsed == True:
            raise MeasurementError(f"The state is now fully collapsed and no more gates can be applied to it")
        elif self.measured_states is not None:
            if gate_location in self.measured_states:
                raise MeasurementError(f"This qubit has been collapsed into a classical state via a previous mesaurement")
        if isinstance(gate_location, int):
            upper_id = Gate.Identity(n=gate_location)   #Gate location starts from Qubit 0
            lower_id = Gate.Identity(n=self.n - gate_location * gate.n - gate.n)        #gate_location * gate.n accounts for the size of the gate applied
        else:
            raise QuantumCircuitError(f"The gate location connot be of {type(gate_location)}, expect type int")
        ndim_gate = upper_id @ gate @ lower_id           #creastes the gate the size of the circuit
        ndim_gate.name = f"{gate.name} on Qubit {gate_location}"
        self.gates.append(ndim_gate)
        if text:
            if ndim_gate.dim < 9:
                print_array(f"Adding this gate to the circuit:")
                print_array(ndim_gate)
            else:
                print_array(f"Adding the {ndim_gate.dim} x {ndim_gate.dim} gate: {ndim_gate.name} to the circuit")

    def compute_final_gate(self, text=True) -> Gate:
        """Combines all gates in the gate list together to produce one unitary final gate"""
        self.final_gate = self.start_gate
        for gate in reversed(self.gates):         #goes backwards through the list and applies them
            self.final_gate = self.final_gate * gate
        self.final_gate.name = f"Final Gate"
        if text:
            print_array(f"The final Gate is:")
            print_array(self.final_gate)
        return self.final_gate
    
    def apply_final_gate(self, text=True) -> Qubit:
        """Applies the final gate to the Quantum state"""
        if self.final_gate is None:
            self.final_gate = self.compute_final_gate()           #applies the function to multiply them all together
        self.state = self.final_gate * self.state
        self.final_gate = None
        if self.noisy:
            self.state = self.add_quantum_channel(self.Q_channel, self.prob)      #creates the noisy state after the gate is applied
            if text:                                                                    #this is a rather crude way to simulate the gate being the thing that
                print_array(f"The final noisy state is:")                                     #introduces errors
                print_array(self.state)
            return self.state
        else:
            if text:
                print_array(f"The final state is:")
                print_array(self.state)
            return self.state

    def list_probs(self, qubit: int=None, povm: np.ndarray=None, text: bool=True) -> Measure:
        """Produces a list of probabilities of measurement outcomes of the state at any point"""
        self.prob_distribution = Measure(state=self.state).list_probs(qubit, povm)            #just lists all of the probabilities of that computed state
        if text:
            print_array(f"The projective probability distribution is:") 
            print_array(format_ket_notation(self.prob_distribution))
        return self.prob_distribution

    def topn_probabilities(self, qubit: int=None, povm: np.ndarray=None, text: bool=True, **kwargs) -> Measure:
        """Only prints or returns a set number of states, mostly useful for large qubit sizes"""
        prob_list = self.list_probs(qubit, povm, text=False)
        topn = kwargs.get("n", 8)
        self.top_prob_dist = top_probs(prob_list, n=topn)                 #purely finds the top probabilities
        if text:
            print_array(f"The top {topn} probabilities are:")
            print_array(format_ket_notation(self.top_prob_dist, type="topn", num_bits=int(np.ceil(np.log2(self.state.dim)))))
        return self.top_prob_dist
    
    def measure_state(self, qubit: int=None, povm: np.ndarray=None, text: bool=True) -> Measure:
        """Measures the state from the list of probabilities"""
        measurement, self.state = Measure(state=self.state).measure_state(qubit, povm)
        if qubit is not None:
            self.measured_states.append(qubit)         #this is to make a list of measured states so that they cant have gates applied
        elif qubit is None:
            self.collapsed = True          #collapses the state if all qubits are measured
        if text:
            if qubit is not None:
                print_array(f"Measured qubit {qubit} as |{measurement}> and the post measurement state is:\n {self.state}")
            else:
                num_bits = int(np.log2(self.state.dim))
                print_array(f"Measured state as |{bin(measurement)[2:].zfill(num_bits)}> and the post measurement state is:\n {self.state}")
        self.gates = []               #wipes the gates to allow for new applications
        return self.state

    def run(self):
        """Can be used to run the whole program on an inital state and set of gates, can be used without this function also"""
        if self.gates == []:
            raise QuantumCircuitError(f"Needs to have gates to be able to apply gates")
        else:
            self.apply_final_gate(text=False)
            self.topn_probabilities(text=False)
            self.measure_state(text=False)
        return self.__rich__()

    def get_von_neumann(self, qubit=None, text=True,):
        """Can compute the Von Neumann entropy for either a single qubit or a whole state"""
        if qubit is None:
            vn_den = Density(state=self.state)
            vne = vn_den.vn_entropy()
            if text:
                print_array(f"Von Neumann entropy of the whole quantum state is {vne}")
            return vne
        elif isinstance(qubit, int):
            if qubit <= self.n and qubit >= 0:
                measure_rho = Measure(state=self.state).list_probs(qubit)[1]
                vn_den = Density(rho=measure_rho)
                vne = vn_den.vn_entropy()
                if text:
                    print_array(f"Von Neumann entropy of qubit {qubit} is {vne}")
                return vne
            raise QuantumCircuitError(f"qubit value must be in the range 0 to {self.n}, not of value {qubit}")
        raise QuantumCircuitError(f"qubit cannot be of type {type(qubit)}, expected int value")

    def get_info(self, attribute, text=True):
        """Mostly used for debugging but can return a specific attribute of the class"""
        if not hasattr(self, attribute): 
            raise QuantumCircuitError(f"This parameter {attribute} of type {type(attribute)} does not exist")
        if text:
            print_array(f"Retrieving the attribute {attribute}:\n {getattr(self, attribute)}")
        return getattr(self, attribute)  
        

class Grover(StrMixin):                                               #this is the Grover algorithms own class
    """The class to run and analyse Grovers algorithm"""
    array_name = "kets"
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
                self.oracle_values.extend(arg)    #chooses between using the given oracle values or finding random ones
            elif isinstance(arg, int):
                self.rand_ov = arg
        
    def phase_oracle(self, qub: Qubit, oracle_values: list) -> Qubit:          #the Grover phase oracle
        """Computes the phase flip produced from the oracle values"""
        qub.vector[oracle_values] *= -1                     #flips the values in the index of the oracle values
        return qub
    
    def optimal_iterations(self, n: int) -> tuple[float, int]:
        """Calculates the best number of iterations for a given number of qubits"""
        search_space: int = 2**n
        op_iter: float = (np.pi/4)*np.sqrt((search_space)/len(self.oracle_values)) -1/2        #the standard equation to find the optimal iterations
        return op_iter, search_space
    
    def init_states(self) -> tuple[Qubit, Gate]:
        """Creates the Quantum state and the Hadamard gates needed for the algorithm"""
        timer = Timer()
        qub = Qubit.q0(n=self.n) #initialises the Qubit state |0*n>
        print_array(f"Initialising state {qub.name}")
        if self.fast:               #this is the faster variant of Grovers utilising the FWHT
            print_array(f"Using FWHT to compute {self.n} x {self.n} Hadamard gate application")
            had = FWHT()                #"creates the had for the FWHT, even thought the FWHT isnt a physical gate"
            return qub, had
        else:
            n_had = Hadamard                   
            print_array(f"Initialising {self.n} x {self.n} Hadamard")
            for i in range(self.n-1):    #creates the qubit and also the tensored hadamard for the given qubit size
                n_had **= Hadamard              #if not fast then tensors up a hadamard to the correct size
                print(f"\r{i+2} x {i+2} Hadamard created", end="")    #allows to clear line without writing a custom print function in print_array
            print(f"\r",end="")
            print_array(f"\rHadamard and Quantum State created, time to create was: {timer.elapsed()[0]:.4f}")
            return qub,n_had

    def iterate_alg(self) -> Qubit:
        """The main core algorithm of the program where the phase gates and Hadamard gates are applied"""
        it = 0
        timer = Timer()
        print_array(f"Running FWHT algorithm:") if self.fast else print_array(f"Running algorithm:")
        qub, had = self.init_states()   #this is where the prveious function is called and the states and hadamard are produced
        while it < int(self.it):   #this is where the bulk of the computation actually occurs and is where the algorithm is actually applied
            print(f"\rIteration {it + 1}:                                                                  ", end="")
            if it != 0:
                qub: Qubit = final_state
            print(f"\rIteration {it + 1}: Applying first Hadamard                                          ", end="")
            hadamard_qubit = had * qub       #applies a hadamard to every qubit                           STEP 1
            print(f"\rIteration {it + 1}: Applying phase oracle                                            ", end="")
            oracle_qubit = self.phase_oracle(hadamard_qubit, self.oracle_values)              #STEP 2   phase flips the given oracle values
            print(f"\rIteration {it + 1}: Applying second Hadamard                                         ", end="")
            intermidary_qubit = had * oracle_qubit                                            #STEP 3 Applies the hadamard again
            print(f"\rIteration {it + 1}: Flipping the Qubits phase except first Qubit                     ", end="")
            intermidary_qubit.vector *= -1           #inverts all of the phases of the qubit values             STEP 4a
            intermidary_qubit.vector[0] *= -1              #inverts back the first qubits phase                 STEP 4b
            print(f"\rIteration {it + 1}: Applying third and final Hadamard                                ", end="")
            final_state = had * intermidary_qubit        #applies yet another hadamard gate to the qubits    STEP 5
            it += 1                   #adds to the iteration counter
            print(f"\r                                                                                     Time elapsed:{timer.elapsed()[0]:.4f} secs", end="")
        print(f"\r",end="")
        print_array(f"\rFinal state calculated. Time to iterate algorithm: {timer.elapsed()[1]:.4f} secs                                                                        ")
        return final_state

    def compute_n(self) -> int:
        """Computes the optimal number of qubits and thus search space by finding a good iteration value based on the specific algorithm ran within"""
        if isinstance(self.n_cap, int):                #check on the input type of self.ncap
            print_array(f"Using up to {self.n_cap} Qubits to run the search")
            max_oracle = max(self.oracle_values)             #finds the largest one to find the min qubits
            n_qubit_min = 1
            while max_oracle > 2**n_qubit_min:             #when picking the qubits, we need enough to allow the search space to be bigger than all the oracle values
                n_qubit_min += 1
            if n_qubit_min > self.n_cap:
                raise QC_error(f"The search space needed for this search is larger than the qubit limit {self.n_cap}.")
            if self.it == None:          #if no given iteration then will find it
                print_array(f"No iteration value given, so will now calculate the optimal iterations")
                n_qubit_range = np.arange(n_qubit_min, self.n_cap + 1, dtype=int)
                if self.iter_calc == None or self.iter_calc == "round":          #this is the default type
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
                                int_val: float = int_dist   #is the standard protocol of rounding to the nearest whole number
                    return self.n
                elif self.iter_calc == "floor":              #is the next protocol that floors every value to the nearest whole number and runs with that
                    int_val = 1
                    print_array(f"Now computing n for the optimal iteration closest to the number below it")
                    for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                        op_iter = self.optimal_iterations(i)[0]                 #computes the optimal amount
                        if op_iter >= 1:
                            int_dist: float = op_iter - np.floor(op_iter)  #finds the float value
                            print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                            if int_dist < int_val:            #iterates through to find the smallest distance from an integer
                                self.n: int = i
                                int_val: float = int_dist
                    return self.n
                elif self.iter_calc == "balanced":        #allows for a balanced approach, this is where if the gap "up" to the next iteration is small enough
                    if isinstance(self.balanced_param, int):            #then is will take that value over going down
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
                        if (1-int_val_round) < int_val_floor / self.balanced_param:      #this is where it calcs which one to use
                            self.n = n_round
                            self.iter_calc = "round"
                            print_array(f"The optimal iteration is computed through rounding")
                        else:
                            self.n = n_floor
                            self.iter_calc = "floor"
                            print_array(f"The optimal iteration is computed by flooring")
                        return self.n
                    raise TypeError(f"balanced_param cannot be of type {type(self.balanced_param)}, expected str")
                raise TypeError(f"iter_calc cannot be of type {type(self.iter_calc)}, expected str")
            else:
                self.n = n_qubit_min
                print_array(f"Running the given {self.it} iterations with the minimum number of qubits {self.n}")
                return self.n
        raise QC_error(f"The qubit limit cannot be of {type(self.n_cap)}, expected type int")
    
    def run(self) -> "Grover":     #Grovers algorithm, can input the number of qubits and also a custom amount of iterations
        """This is the function to initiate the search and compiles the other functions together and prints the values out"""
        Grover_timer = Timer()
        if self.rand_ov:
            console.rule(f"Grovers search with random oracle values", style="grover_header")
            self.oracle_values = np.ones(self.rand_ov)    #basically if oraclae values are not given, it will find n random values and use those instead
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
            raise QC_error(f"self.n is of the wrong type {type(self.n)}, expected type int")

        if self.rand_ov:
            self.oracle_values = []
            for i in range(self.rand_ov):
                self.oracle_values.append(randint(0, 2**self.n - 1))
            self.rand_ov = self.oracle_values   #computes the random oracle values and replaces oracle values with them
        op_iter = self.optimal_iterations(self.n)[0]
        if self.it == None:     #now picks an iteration value
            if self.iter_calc == "round" or self.iter_calc == None:
                self.it = round(op_iter)
            elif self.iter_calc == "floor":
                self.it = int(np.floor(op_iter))      #picks which algorithm to apply
            elif self.iter_calc == "balanced":
                self.it = int(np.floor(op_iter))
            else:
                raise QC_error(f"Invalid keyword argument {self.iter_calc} of {type(self.iter_calc)}")
            
            if self.it < 1:    #obviously we cant have no iterations so it atleast does 1 iteration
                self.it = 1.0       #more of a failsafe, will almost certainly be wrong as the percentage diff from the it value to 1 will be large?
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
        sorted_arr = top_probs(final_state.list_probs(), n=len(self.oracle_values))         #finds the n top probabilities
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
    """Custom print function to neatly arrange matrices and also print with a nice font"""
    def __init__(self, array):
        self.array = array
        self.prec = 3  # Default precision for numpy formatting
        np.set_printoptions(
            precision=self.prec,
            suppress=True,
            floatmode="fixed")
        if isinstance(array, Measure):
            console.print(array,markup=True, style="measure")
        elif isinstance(array, Density):
            console.print(array,markup=True,style="density")
        elif isinstance(array, Gate):
            console.print(array,markup=True,style="gate")
        elif isinstance(array, Grover):
            console.print(array,markup=True, style="prob_dist")
        elif isinstance(array, Qubit):
            console.print(array,markup=True,style="qubit")
        elif isinstance(array, np.ndarray):
            console.print(array, markup=True, style="gate")
        else:
            console.print(array,markup=True,style="info")
        


oracle_values = [9,4,3,2,5,6,12,15,16]
oracle_values2 = [1,2,3,4,664,77,5,10,12,14,16,333,334,335,400,401,41,42,1000]
oracle_values3 = [1,10]
oracle_values4 = [500,5,4,7,8,9,99]
oracle_value_test = [1,2,3]
large_oracle_values = [1120,2005,3003,4010,5000,6047,7023,8067,9098,10000,11089,12090,13074]

def main():
    """Where you can run commands without it affecting programs that import this program"""
    print_array(q0 @ q0)
    print_array(Hadamard + Hadamard)
    print_array(Hadamard - Hadamard)
    testp = Density(state=qp)
    testm = Density(state = qm)
    testp2 = Density(state=qp)
    print_array(testp + testm)
    print_array(testp - testm)

    print_array(trace(CNot))
    print_array(testp == testp2)
    M_test1 = Measure(state=qm @ qp)
    M_test2 = Measure(state=qm @ qp)
    print_array(M_test1 == M_test2)
    print_array(M_test1.probs)
    print_array(Density(state_a = q1, state_b = qpi).fidelity())
    test = Circuit(n=1, state=qp, noisy=True, Q_channel="P flip", prob=0.5)
    test.get_info("state")
    test.apply_final_gate()
    test.list_probs()
    Grover(8).run()
    print_array(Hadamard)
    print_array(Hadamard @ Hadamard)
    print_array(Density(state=qpi) @ Density(state=qmi))
    Grover([1], n=3, iter_calc="floor").run()
    print_array(Qubit(type="seperable", vectors=[q0,q0,q1]))
    se_test = Qubit(type="mixed", vectors=[q0,q1], weights=[0.2,0.8],detailed=True)
    print_array(se_test)
    print_array(se_test.density)
    print_array(se_test.se)
    se_test = Qubit(type="mixed", vectors=[[1,0],[0,1]], weights=[0.2,0.8],detailed=True)
    print_array(se_test)
    print_array(se_test.density)
    print_array(se_test.se)
    print_array(Identity @ Identity)