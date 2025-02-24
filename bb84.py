print("Script is running...")
import time
start = time.time()
import numpy as np                            #mostly used to make 1D arrays
import random as rm                                       #used for measuring
import atexit
import matplotlib.pyplot as plt
from rich.console import Console
from rich.theme import Theme
import cProfile

# I was thinking of making a little game of BB84 where you play different
#   characters to both test and understand the scheme.
# We could pull from the ChatGPT API to create fully encrypted automated
#   conversations.
# the basic idea I have is that you could choose to play Alice, Bob, or Eve
# BOB:
#   comes up with a message of bitlength n that can be answered by a one bit
#       response 
#   thus, wants a final key length of atleast n+1 (Alice takes first n, Bob
#       takes final one)
#   randomly generates a key of length k and sends it to Bob (should be such
#       that (n+1)/((1-f)k) \leq 1/2) where f is the fraction to be shared 
#       publicly
#   Alice holds their (x,a) tuples, publicizes their x list
#   see's the y list from Bob and ignores those qubits where x != y
#   thus we should have a = b for the rest of the list and the list should be
#       atleast of length (n+1)/(1-f)
#   now, randomly choose f*k of the bits and share them publicly
#   if the error rate if over 11% we discard
#   otherwise, we take the remaining list k' and verify k' \geq n+1
#   if so, we XOR our message with the first n bits
#   Bob randomly chooses a bit (automated response) and XORs it with the final
#       bit in k' as a response
#   Alice then decodes this message by a second XOR and get's their response
#   NOTE: I often talk about bits in the key (like in k or k'), but I really
#   mean bits in the associated a and b lists
#
# BOB:
#   Alice randomly comes up with a message of bitlength n that can be answered
#       by a one bit response
#   thus, want a final key length of atleast n+1 (Alice takes first n, Bob
#       takes final one)
#   Alice randomly generates a key of length k and sends it to Bob (should be
#       such that (n+1)/((1-f)k) \leq 1/2) where f is the fraction to be shared 
#       publicly
#   Bob randomly creates their y list and measures against k to get their b list
#   Alice holds their (x,a) list, publicizes their x's, Bob does too for (y,b)
#   Bob sees Alice's x list and ignores those qubits where x != y
#   thus we should have a = b for the rest of the list and the list should be
#       atleast of length (n+1)/(1-f)
#   now, randomly choose f*k of the bits and share them publicly
#   if the error rate if over 11% we discard
#   otherwise, we take the remaining list k' and verify k' \geq n+1
#   now, we recieve Alice's message and XOR it with the first n digits of k'
#   Bob chooses a response and XOR's it with the last bit of k' and sends it
#   Thus we get to answer questions securely!
#
# EVE:
#   Get's to try and steal the message without getting caught.

# FIRST: WE HAVE OUR GENERAL QUBIT CLASS
class Qubit:              
    def __init__(self, **kwargs) -> None:
        self.state_type = kwargs.get("type", "pure") #HM: if a value for the key type isn't passed into the Qubit definition, it defaults to "pure"
        self.name = kwargs.get("name","|0>")
        self.vector = np.array(kwargs.get("vector",np.array([1,0])),dtype=np.complex128) #HM: isn't this redundant? do we want an array of an array?
        self.dim = len(self.vector)                    #used constantly in all calcs so defined it universally
        self.shift = self.dim.bit_length() - 1 #HM: this returns the # of bits required to represent self.dim minus 1
        self.density = None
        if self.state_type == "pure":
            pass
        elif self.state_type == "mixed":
            self.vector = np.array(kwargs.get("vectors",[]),dtype = np.complex128)
            self.weights = kwargs.get("weights", [])
            self.density = self.density_mat()
        elif self.state_type == "seperable":
            qubit_states = np.array(kwargs.get("vectors",[]),dtype = np.complex128)
            self.vector = qubit_states[0]
            for state in qubit_states[1:]:
                self.vector = self @ state #HM: @ here is mat multiplication
        elif self.state_type == "entangled":
            pass
        else:
            pass #HM: maybe worth throwing an error of some sorts here (unless it's covered elsewhere)

    @classmethod
    def q0(cls):
        q0_vector = [1,0]
        return cls(name="|0>", vector=q0_vector)

    @classmethod
    def q1(cls):
        q1_vector = [0,1]
        return cls(name="|1>", vector=q1_vector)

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
        

    def __str__(self):
        return f"{self.name}\n{self.vector}"   #did this so that the matrix prints neatly
    
    def __rich__(self):
        return f"[bold]{self.name}[/bold]\n[not bold]{self.vector}[/not bold]"
    
    def __matmul__(self, other):               #this is an n x n tensor product function
        if isinstance(other, Qubit):           #although this tensors are all 1D   
            self_name_size = int(np.log2(self.dim))
            other_name_size = int(np.log2(other.dim))
            new_name = f"|{self.name[1:self_name_size+1]}{other.name[1:other_name_size+1]}>" #HM: this indexing seems wrong?
            new_length: int = self.dim*other.dim #HM: so this is a tensor multiplication in this case?
            new_vector = np.zeros(new_length,dtype=np.complex128)
            other_shift = other.dim.bit_length() - 1
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other.dim):          #iterates up and down the second ket
                    new_vector[j+(i << other_shift)] += self.vector[i]*other.vector[j] #adds the values into
            return Qubit(name=new_name, vector=np.array(new_vector))    #returns a new Object with a new name too
        elif isinstance(other, np.ndarray): #HM: what other objects are we possibly even passing through?
            other_dim = len(other)
            self_name_size = int(np.log2(self.dim)) # the int() wrapper truncates to the int part (like a floor function for postive numbers)
            other_name_size = int(np.log2(other_dim))
            new_name = f""
            new_length: int = self.dim*other_dim
            new_vector = np.zeros(new_length,dtype=np.complex128)
            for i in range(self.dim):     #multiplies the second ket by each value in the first ket
                for j in range(other_dim):          #iterates up and down the second ket
                    new_vector[j+(i * other_dim)] += self.vector[i]*other[j] #adds the values into
            self.dim = new_length
            return np.array(new_vector)    #returns a new Object with a new name too

            print(test)
        else:
            raise QC_error(qc_dat.error_class)

    #HM: this seems to be a Kronecker product
    def __ipow__(self, other):                 #denoted **=
        if isinstance(self, Qubit):  
            self = self @ other
            return self
        elif isinstance(self, Qubit.vector):
            print(test2)
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

    def prob_dist(self, final_gate=None):            #creates a table with the prob of each state occuring, only works for projective measurements
        new_mat = np.zeros(self.dim,dtype=np.float64)
        if isinstance(self, Qubit):
            norm = 0
            if final_gate:
                if isinstance(final_gate, Gate):
                    new_name = f"PD for {self.name} applied to Circuit:"
                    new_state = final_gate * self             #creates the new state vector
                    state_conj = np.conj(new_state.vector)
                    for i in range(self.dim):             #basically squares the vector components to get the prob of each
                        new_mat[i] = (new_state.vector[i]*state_conj[i]).real
                        norm += new_mat[i]
                else:
                    raise QC_error(qc_dat.error_class)
            else:
                qubit_conj = np.conj(self.vector)
                new_name = f"PD for {self.name}"
                for i in range(self.dim):
                    new_mat[i] = (self.vector[i]*qubit_conj[i]).real        #just does the squaring without any external matrices applied
                    norm += new_mat[i]
            if np.isclose(norm, 1.0, atol=1e-5):
                return Prob_dist(new_name, qc_dat.prob_dist_info, np.array(new_mat))
            else:
                raise QC_error(qc_dat.error_norm)
        else:
            raise QC_error(qc_dat.error_class)

    def measure(self, final_gate=None):         #randomly picks a state from the weighted probabilities, can also apply the gate within it, which is a bit redundant
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

# SECOND: WE ARE GOING TO MAKE A SIMPLE BOB-CENTERED IMPLEMENTATION
#   okay we want Bob to be able to read-in messages randomly generate a key

def m_to_bits(m):
    return ''.join(format(ord(c), '08b') for c in m)

def bits_to_m(bit_m):
    chars = [bit_m[i:i+8] for i in range(0, len(), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

class BOB:
    def __init__(self, message, fraction, error_tolerance):
        self.message = message
        self.fraction = fraction
        self.error_tolerance = error_tolerance
        self.quantum_key = []
        self.x = []
        self.a = []
    
    def generate_big_key(self):
        big_key_length = (len(m_to_bits(self.message)) + 1)*3
        # I haven't implemented fraction here yet, just using *3 for safety
        for i in range(big_key_length):
            r_num = rm.randint(0,3)
            if r_num == 0:
                self.quantum_key.append(Qubit.q0)
                self.x.append(0)
                self.a.append(0)
            elif r_num == 1:
                self.quantum_key.append(Qubit.q1)
                self.x.append(0)
                self.a.append(1)
            elif r_num == 2:
                self.quantum_key.append(Qubit.qp)
                self.x.append(1)
                self.a.append(0)
            elif r_num == 3:
                self.quantum_key.append(Qubit.qm)
                self.x.append(0)
                self.a.append(1)
    
    def prune(self, l):
        for element in l:
            del self.y[element]
            del self.b[element]

    def encode(self):
        short_key = self.a[:len(self.message)]
        final_key = ''.join(map(str, short_key))
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        # two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        return bin(int(m_to_bits(self.message),2) ^ int(final_key, 2))[2:].zfill(len(self.message))
    def decode(self, message):
        short_key = self.a[len(self.a) - 1]
        if bin(short_key ^ message)[2:] == 0:
            return "NO"
        elif bin(short_key ^ message)[2:] == 1:
            return "YES"

class ALICE:
    def __init__(self):
        self.response = None
        self.y = []
        self.b = []
        self.answer = rm.randint(0,1)

    def create_y(self, l):
        for i in range(l):
            r_num = rm.randint(0,1)
            self.y.append(r_num)

    def measure_key(self, key):
        if len(self.y) != len(key):
            raise ValueError("Key and measurement device length mismatch.")
        for i in range(len(self.y)):
            self.b.append(key[i].measure(self.y[i]))
            # this I have yet to impliment, needs to be able to measure our
            # qubit in the desired basis
    def prune(self, l):
        for element in l:
            del self.y[element]
            del self.b[element]
    
    def decode(self, code):
        short_key = self.b[:len(code)]
        final_key = ''.join(map(str, short_key))
        bin_decoding = bin(int(m_to_bits(self.code),2) ^ int(final_key, 2))[2:].zfill(len(self.code))
        return bits_to_m(bin_decoding)
    
    def answer(self):
        return rm.randint(0,1)
    


def play():
    user_input = input("Hi Bob, enter your secure message please:")
    fraction = float(input("Thank you. Now, please input the fraction of bits you want to use to check for error:"))
    while fraction >= 1 or fraction < 0:
        fraction = float(input("Error checking fraction must be in [0,1), please try again:"))
    error_tolerance = float(input("Now, please describe your error tolerance:"))
    while error_tolerance >= 1 or error_tolerance < 0:
        error_tolerance = float(input("Error tolerance must be in [0,1), please try again:"))
    Bob = BOB(user_input,fraction,error_tolerance)
    Bob.generate_big_key()
    Alice = ALICE()
    Alice.create_y(len(m_to_bits(user_input)))
    Alice.measure_key(Bob.quantum_key)
    if len(Bob.x) != len(Alice.y): raise ValueError("Bob and Alice length mismatch.")
    bad_indices = []
    for i in range(len(Bob.x)):
        if Bob.x[i] != Alice.y[i]: bad_indices.append(i)
    Bob.prune(bad_indices)
    Alice.prunce(bad_indices)
    code = Bob.encode()
    print("Your message has succesfully been encoded.")
    message_alice_recieves = Alice.decode(Bob.encode())
    print("We have received and decoded Alice's message. The message:" + Bob.decode(Alice.answer()))

    

play()
print("Script completed run.")