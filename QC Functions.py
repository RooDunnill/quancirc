import numpy as np                                                            #mostly used to make 1D arrays
import sympy as sp                                              #mostly used for sqrt function, and simplify
import random as rm                                             #used for measuring
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

sp.init_printing(use_unicode=True)                              #pretty much useless i think
plot_counter = 0
timer_switch = 1            #wanted to be able to turn it off after testing is done but not remove from code fully
class timer:
    def __init__(self, state):
        self.state = state

    def timer(initial):      
        global start_time, stop_time, interval
        if timer_switch == 1:
            if initial == 1:
                start_time = time.time()
            elif initial == 0:
                stop_time = time.time()
                interval = (stop_time - start_time)
                interval = "{:.3f}".format(interval)        #makes it a little neater
                stop_time = 0
                start_time = 0
                print("Time taken to run entire program is: " + interval)
        elif timer_switch == 0:
            pass
timer.timer(1)

class Gate_data:                    #defines a class to store variables in to recall from so that its all
    C_Not_info = """This gate is used to change the behaviour of one qubit based on another. 
    This sepecific function is mostly obselete now, it is preferred to use the C Gate class instead"""       #C_Not is mostly obsolete due to new C Gate class       #in one neat area
    C_Not_matrix = [1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0]#2 Qubit CNot gate in one configuration, need to add more
    X_Gate_info = "Used to flip the Qubit in the X basis. Often seen in the CNot gate."             
    X_matrix = [0,1,1,0]                           
    Y_Gate_info = "Used to flip the Qubit in the Y basis."
    Y_matrix = [0,np.complex128(0-1j),np.complex128(0+1j),0]           
    Z_Gate_info = "Used to flip the Qubit in the Z basis. This gate flips from 1 to 0 in the computational basis."
    Z_matrix = [1,0,0,-1]                                      
    n = 1/sp.sqrt(2)
    Hadamard_info = """The Hadamard gate is one of the most useful gates, used to convert the Qubit from
    the computation basis to the plus, minus basis. When applied twice, it can lead back to its original value/
    acts as an Identity matrix."""
    Hadamard_matrix = [n,n,n,-n]
    U_Gate_info = """This is a gate that can be transformed into most elementary gates using the constants a,b and c.
    For example a Hadamard gate can be defined with a = pi, b = 0 and c = pi while an X Gate can be defined by
     a = pi/2 b = 0 and c = pi. """
    Identity_info = """
    Identity Matrix: This matrix leaves the product invariant after multiplication.
    It is mainly used in this program to increase the dimension
    of other matrices. This is used within the tensor products when
    a Qubit has no gate action, but the others do.
"""
    Identity_matrix = [1,0,0,1]
    error_class = "the operation is not with the correct class"
    error_mat_dim = "the dimensions of the matrices do not share the same value"
    error_value = "the value selected is outside the correct range of options"
    error_qubit_num = "you can't select the same qubit for both inputs of the operation gate and control gate"
    error_qubit_pos = "one of the selected qubits must be 1 which represents the top left of the matrix rather than qubit 1"
    Density_matrix_info = "test"
    error_trace = "the trace does not equal 1 and so the calculation has gone wrong somewhere"
    


def trace(matrix):
    if isinstance(matrix, Gate):
        tr = 0
        for i in range(matrix.dim-1):
            tr += matrix.matrix[i+i*matrix.dim]
        return tr
    else:
        print(Gate_data.error_class)

class Qubit:                        
    def __init__(self, name, vector):
        self.name = name
        self.vector = np.array(vector,dtype=np.complex128)
        self.dim = len(vector)                    #used constantly in all calcs so defined it universally
        
    def __str__(self):
        return f"{self.name}\n{self.vector}"   #did this so that the matrix prints neatly
    
    def __matmul__(self, other):               #this is an n x n tensor product function
        if isinstance(other, Qubit):           #although this tensors are all 1D
            new_name = f"{self.name} @ {other.name}"     #no tensor symbol so this will do
            new_length = self.dim*other.dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other.dim):          #iterates up and down the second ket
                    new_vector[j+i*other.dim] += self.vector[i]*other.vector[j] #adds the values into
            sp.simplify(new_vector)                           #a new matrix in order
            return Qubit(new_name, np.array(new_vector))    #returns a new Object with a new name too
        else:
            print(Gate_data.error_class)

    def norm(self):                 #dunno why this is here ngl, just one of the first functions i tried
        normalise = sp.sqrt(sum([i*np.conj(i) for i in self.vector]))
        self.vector = sp.simplify(self.vector/normalise)

    def qubit_info(self):      
        print("""The Qubit is the quantum equivalent to the bit. However, due to the nature of 
        Quantum Mechanics, it can take any value rather than just two. However, by measuring the state
        in which it is in, you collapse the wavefunction and the Qubit becomes 1 of two values, 1 or 0.""")

    

    def density_mat(self):
        new_name =f"Density matrix of qubit {self.name}"
        new_mat = np.zeros(self.dim*self.dim,dtype=np.complex128)
        qubit_conj = np.conj(self.vector)
        for i in range(self.dim):
            for j in range(self.dim):
                new_mat[j+i*self.dim] += qubit_conj[i]*self.vector[j]
        return Density(new_name, Gate_data.Density_matrix_info, new_mat)

    def prob_state(self, meas_state, final_gate):  #this is just flat out wrong atm p(i) = Tr[Pi rho Pi+]
        if isinstance(self and meas_state, Qubit)  and isinstance(final_gate, Gate):
            projector = meas_state.density_mat()
            final_state = final_gate * self
            den = final_state.density_mat()
            probability = trace(projector * den)
            is_real = np.isreal(probability)
            if is_real is True:
                return probability
            else:
                print("test imag error")
        else:
            print(Gate_data.error_class)

    def prob_dist(self, final_gate):
        if isinstance(self, Qubit) and isinstance(final_gate, Gate):
            new_mat = np.zeros(self.dim,dtype=np.float64)
            for i in range(self.dim):
                meas_state_vector = np.zeros(self.dim,dtype=np.complex128)
                meas_state_vector[i] += 1
                meas_state = Qubit("f|{i}",meas_state_vector)
                new_mat[i] = self.prob_state(meas_state, final_gate)
            return new_mat

    def bloch_plot(self, qubit):
        global plot_counter, ax
        if qubit <= self.dim:
            vals = np.zeros(2,dtype=np.complex128)
            vals[0] = self.vector[2*qubit-2]
            vals[1] = self.vector[2*qubit-1]
            plotted_qubit = Qubit("", vals)
            print(vals)
            den_mat = plotted_qubit.density_mat()
            x = 2*np.real(den_mat.matrix[1])
            y = 2*np.imag(den_mat.matrix[2])
            z = den_mat.matrix[0] - den_mat.matrix[3]
            if plot_counter == 0:
                ax = plt.axes(projection="3d")
            ax.quiver(0,0,0,x,y,z)
            plot_counter += 1
        else:
            print(Gate_data.error_value)
q0_matrix = [1,0]
q1_matrix = [0,1]
qplus_matrix = [1,1]
qminus_matrix = [1,-1]
q0 = Qubit("q0",q0_matrix)
q1 = Qubit("q1",q1_matrix)
qplus = Qubit("q+",qplus_matrix)
qminus = Qubit("q-",qminus_matrix)
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
        self.dim = int(sp.sqrt(self.length))

    def __str__(self):
        return f"{self.name}\n{self.matrix}"

    def __matmul__(self, other):
        if isinstance(other, Gate):
            new_info = "This is a tensor product of gates: "f"{self.name}"" and "f"{other.name}"
            new_name = f"{self.name} @ {other.name}"
            new_length = self.length*other.length
            new_dim = sp.sqrt(new_length)
            new_mat = np.zeros(new_length,dtype=np.complex128)
            for m in range(self.dim):
                for i in range(self.dim):
                    for j in range(other.dim):
                        for k in range(other.dim):   #honestly, this works but is trash and looks like shit
                            new_mat[k+j*new_dim+other.dim*i+other.dim*new_dim*m] += self.matrix[i+self.dim*m]*other.matrix[k+other.dim*j]
            sp.simplify(new_mat)                     #will try to impliment a XOR operation for this which should be a lot faster
            return Gate(new_name, new_info, np.array(new_mat))
        else:
            print(Gate_data.error_class)

    def __mul__(self, other):       #matrix multiplication
        summ = np.zeros(1,dtype=np.complex128)  #could delete summ and make more elegant
        if isinstance(other, Gate):    #however probs completely better way to do this so might scrap at some point
            if self.dim == other.dim:
                new_info = "This is a matrix multiplication of gates: "f"{self.name}"" and "f"{other.name}"
                new_name = f"{self.name} * {other.name}"
                new_mat = np.zeros(self.length,dtype=np.complex128)
                for i in range(self.dim):
                    for k in range(self.dim):
                        for j in range(self.dim):    #again a mess and done in a different manner to tensor product
                            summ[0] += (self.matrix[j+self.dim*i]*other.matrix[k+j*self.dim])
                        new_mat[k+self.dim*i] += summ[0]
                        summ = np.zeros(1,dtype=np.complex128)
                sp.simplify(new_mat)
                if isinstance(other, Density):
                    new_info = "This is the density matrix of: "f"{self.name}"" and "f"{other.name}"
                    return Density(new_name, new_info, new_mat)
                return Gate(new_name, new_info, np.array(new_mat))
            else:
                print(Gate_data.error_mat_dim)
        elif isinstance(other, Qubit):  #splits up based on type as this isnt two n x n but rather n x n and n matrix
            if self.dim == other.dim:
                new_name = f"[{self.name}] {other.name}"
                new_mat = np.zeros(self.dim,dtype=np.complex128)
                for i in range(self.dim):
                        for j in range(self.dim):
                            summ[0] += (self.matrix[j+self.dim*i]*other.vector[j])
                        new_mat[i] += summ[0]
                        summ = np.zeros(1,dtype=np.complex128)
                return Qubit(new_name, np.array(new_mat))
            else:
                print(Gate_data.error_mat_dim)
        else:
            print(Gate_data.error_class)
    
    def __add__(self, other):         #direct sum
        if isinstance(other, Gate):
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
            print(Gate_data.error_class)
    
    def __iadd__(self, other):
        if isinstance(other, Gate):
            self = self + other
            return self
        else:
            print(Gate_data.error_class)
        
        

    def gate_info(self):
        print(
    """Gates are used to apply an operation to a Qubit.
    They are normally situated on a grid of n Qubits.
    Using tensor products, we can combine all the gates 
    at one time instance together to create one unitary matrix.
    Then we can matrix multiply successive gates together to creat one
    universal matrix that we can apply to the Qubit before measuring""")
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
                print(Gate_data.error_qubit_num)
        else:
            print(Gate_data.error_qubit_pos)
        self.matrix = new_mat.matrix
        self.dim = int(abs(qubit_dist)*Identity.dim+gate_action.dim)
        self.length = self.dim*self.dim
                

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
        self.dim = int(sp.sqrt(self.length))
        
class Density(Gate):
    def __init__(self, name, info, matrix):
        self.name = name
        self.info = info
        self.matrix = matrix
        self.length = len(self.matrix)
        self.dim = int(sp.sqrt(self.length))

    
        

    
class print_array:    #made to try to make matrices look prettier
    def __init__(self, array):         #probs could have used sp.pretty or whatever but didnt wanna confuse
        self.array = array             #sp and np matrices and get confused
        prec = 3
        if isinstance(array, Qubit):
            np.set_printoptions(precision=prec,linewidth=20,suppress=True,floatmode="fixed")
            print(array)
        elif isinstance(array, Gate):           #so janky
            np.set_printoptions(precision=prec,linewidth=(2+2*(3+prec))*array.dim,suppress=True,floatmode="fixed")
            print(array)
        else:
            print(Gate_data.error_class)


X_Gate = Gate("X", Gate_data.X_Gate_info, Gate_data.X_matrix)
Y_Gate = Gate("Y",Gate_data.Y_Gate_info, Gate_data.Y_matrix)
Z_Gate = Gate("Z",Gate_data.Z_Gate_info, Gate_data.Z_matrix)
Identity = Gate("I",Gate_data.Identity_info, Gate_data.Identity_matrix)
Hadamard = Gate("H",Gate_data.Hadamard_info, Gate_data.Hadamard_matrix)
U_Gate_X = U_Gate("Universal X", Gate_data.U_Gate_info, np.pi, 0, np.pi)
U_Gate_H = U_Gate("Universal H", Gate_data.U_Gate_info, np.pi/2, 0, np.pi)
CNot_flip = C_Gate("CNot", Gate_data.C_Not_matrix, X_Gate, 2, 1)
CNot = C_Gate("CNot", Gate_data.C_Not_matrix, X_Gate, 1, 2)
def Test_Alg(Qubit):         #make sure to mat mult the correct order
    gate1 = X_Gate @ CNot
    gate2 = Hadamard @ Hadamard @ X_Gate
    gate3 = CNot @ X_Gate
    gate4 = CNot @ Hadamard
    alg = gate4 * gate3 * gate2 * gate1
    result = alg * Qubit
    return result
test= q0.density_mat()
test2 = Hadamard * test

meas_state = q0 @ q0 @ q0
qub = q1 @ q1 @ q0
gate_1 = X_Gate @ X_Gate @ X_Gate
gate_2 = Hadamard @ Y_Gate @ Z_Gate
gate_final = gate_2 * gate_1
print_array(gate_final)
print(qub.prob_state(meas_state, gate_final))
print(qub.prob_dist(gate_final))

timer.timer(0)
if plot_counter > 0:
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_sp = np.cos(u)*np.sin(v)
    y_sp = np.sin(u)*np.sin(v)
    z_sp = np.cos(v)
    ax.set_xlabel("X_Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Bloch Sphere")
    ax.plot_surface(x_sp, y_sp, z_sp, color="g", alpha=0.3)
    ax.legend()
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
    plt.show()