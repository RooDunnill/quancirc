print("Script is running...")

import numpy as np
import random as rm
import qcp_program
import tkinter as tk

# CHECK IF KEY_LENGTH IS FULLY CONSISTENT
# CHECK TO ENSURE FDECODE IS WORKING (SHOULD ALWAYS BE CORRECT)
# could also factor fraction into key_length decision

############################### BACKGROUND INFO ###############################
# HOW OUR GAME WORKS:
#
# will write later
#
# How B92 Works:
# Alice sends qubits from the set {|0>,|->} and creates a list a^i
# Bob randomly chooses wither Comp or H basis and makes a list y^i
# Bob measures and makes a list b^i putting down 0 whenever he gets |+>
#   and putting down 1 whenever he gets |1>
# He shares the positions in the quantum stream by which he found these states
#   with Alice, and they only use those positions
# For those positions, a^i = b^i \forall i
# We then can do public checks for QBER etc
# DOWNSIDES: Lower noise tolerance and rate
# UPSIDES: Simpler implementation

# How BBM92 Works (enganglement based BB84)
# Any trusted or untrusted 3rd party (even Eve) distributes n copies of the
#   state |\Phi> = \sqrt(1/2) (|00> + |11>) = \sqrt(1/2)(|++> + |-->)
# Alice and Bob each randomly select a basis and measure the result
# They then publicly share which bases they are using and only keep the results
#   from when the basis are the same. The resulting a^i 's and b^i 's should
#   be the same (you then do public chekcs for QBER etc)
#
# To an adversary I think this looks pretty much the same as BB84
# UPSIDES: It makes security proof easier. It allows a third party that may or
#   may not be trusted to prepare state and users only have to measure them.
# DOWNSIDES: It is harder to prepare entangled states and share them then
#   single qubit states.
###############################################################################

# takes a string, and converts to a bitstring
def m_to_bits(m):
    return ''.join(format(ord(c), '08b') for c in m)

# takes a bitstring and converts back to a string (the inverse of above)
def bits_to_m(bit_m):
    if len(bit_m) % 8 != 0:
        raise ValueError("Input bit string length must be a multiple of 8")
    chars = [bit_m[i:i+8] for i in range(0, len(bit_m), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

# this class creates objects of type Alice (first communicator)
# self.a is the classical key we create
class ALICE:
    def __init__(self, message, fraction, error_tolerance):
        self.message = message # message to share with Bob
        self.fraction = fraction # fraction of bits to use to check for errors
        self.error_tolerance = error_tolerance # maximum error % we tolerate
        self.quantum_key = [] # the quantum key that Alice generates
        self.x = [] # the list of bases used in creating quantum_key
        self.a = [] # list of choice of qubit within given basis
                    #   for example: |0>,|+> are 0, while |1>,|-> are 1
    
    # method to generate the quantum key
    def generate_big_key(self, protocol, key_length):
        if protocol == 0: # BB84
            for i in range(key_length):
                r_num = rm.randint(0,3)
                if r_num == 0:
                    self.quantum_key.append(qcp_program.Qubit.q0())
                    self.x.append(0)
                    self.a.append(0)
                elif r_num == 1:
                    self.quantum_key.append(qcp_program.Qubit.q1())
                    self.x.append(0)
                    self.a.append(1)
                elif r_num == 2:
                    self.quantum_key.append(qcp_program.Qubit.qp())
                    self.x.append(1)
                    self.a.append(0)
                elif r_num == 3:
                    self.quantum_key.append(qcp_program.Qubit.qm())
                    self.x.append(1)
                    self.a.append(1)
        elif protocol == 1: # six-state (SS) QKD
            for i in range(key_length):
                r_num = rm.randint(0,5)
                if r_num == 0:
                    self.quantum_key.append(qcp_program.Qubit.q0())
                    self.x.append(0)
                    self.a.append(0)
                elif r_num == 1:
                    self.quantum_key.append(qcp_program.Qubit.q1())
                    self.x.append(0)
                    self.a.append(1)
                elif r_num == 2:
                    self.quantum_key.append(qcp_program.Qubit.qp())
                    self.x.append(1)
                    self.a.append(0)
                elif r_num == 3:
                    self.quantum_key.append(qcp_program.Qubit.qm())
                    self.x.append(1)
                    self.a.append(1)
                elif r_num == 4:
                    self.quantum_key.append(qcp_program.Qubit.qpi())
                    self.x.append(2)
                    self.a.append(0)
                elif r_num == 5:
                    self.quantum_key.append(qcp_program.Qubit.qmi())
                    self.x.append(2)
                    self.a.append(1)
        elif protocol == 2: #B92
            for i in range(key_length):
                r_num = rm.randint(0,1)
                if r_num == 0:
                    self.quantum_key.append(qcp_program.Qubit.q0())
                    self.a.append(0)
                elif r_num == 1:
                    self.quantum_key.append(qcp_program.Qubit.qm())
                    self.a.append(1)
        else: raise ValueError("We aren't using any of our protocols")
        # BBM92 is generated by EVE

    # creates x-list (basis list) randomly, only used in BBM92
    def create_x(self, key_length):
        for _ in range(key_length):
            r_num = rm.randint(0,1)
            self.x.append(r_num)
    
    # used for removing x and a elements that don't align with Bob
    def prune(self, l):
        for element in l:
            if protocol != 2: del self.x[element] #no basis choice in B92
            del self.a[element]

    # returns our cyphertext (the message encoded with our key)
    def encode(self):
        bit_m = m_to_bits(self.message)
        # short_key grabs enough of the key to apply to our message
        short_key = [self.a.pop(0) for _ in range(len(bit_m))]
        # if the key and message length aren't equal we cant use one-time pad
        if len(bit_m) != len(short_key):
            raise ValueError("message bits and key are of different length")
        final_key = ''.join(map(str, short_key)) # converts list to bitstring
        # cypher bitwise XOR's the message and key
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        #   two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        cypher = bin(int(bit_m,2) ^ int(final_key,2))[2:].zfill(len(bit_m))
        return cypher
    
    # decodes Bob's message and converst to a string
    def decode(self, recieved_message):
        l = len(recieved_message)
        # short_key grabs enough of the key to apply to our message
        short_key = [self.a.pop(0) for _ in range(l)]
        final_key = ''.join(map(str, short_key)) # converts list to bitstring
        # bit_m bitwise XOR's the recieved_message and key
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        #   two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        bit_m = bin(int(recieved_message,2) ^ int(final_key,2))[2:].zfill(len(recieved_message))
        return bits_to_m(bit_m) # returns string form of recieved_message

# creates objects of type Bob (second commmunicator)
# self.b is the classical key we create
class BOB:
    def __init__(self):
        self.y = [] # list of bases used when measuring quantum key
        self.b = [] # list of results (0 or 1) when using self.y on quantum key

    # randomly creates lits of bases self.y
    def create_y(self, l, protocol):
        if protocol == 1: # six-state (SS) QKD
            for i in range(l):
                r_num = rm.randint(0,2)
                self.y.append(r_num)
        elif protocol == 0 or protocol == 2 or protocol == 3:
            for i in range(l):
                r_num = rm.randint(0,1)
                self.y.append(r_num)

    # creates self.b list by measure key with self.y basis list
    def measure_key(self, key):
        if len(self.y) != len(key): # error checking
            print("y length: "+ str(len(self.y)))
            print("key length: " + str(len(key)))
            raise ValueError("Key and measurement device length mismatch.")
        for i in range(len(self.y)):
            qbit = key[i]
            if self.y[i] == 0: pass # 0 means computational basis
            elif self.y[i] == 1: # 1 means Hadamard basis
                H = qcp_program.Gate.Hadamard() # Hadamard gate
                qbit = H.__mul__(qbit) # apply Hadamard gate to qbit
            elif self.y[i] == 2: # 2 means third basis for SS
                H = qcp_program.Gate.Hadamard()
                P = qcp_program.Gate.P_Gate(-np.pi/2) # creates S^adjoint
                qbit = H.__mul__(P.__mul__(qbit)) # applies HS^* to qbit
            # we can now measure in computational basis as we have already
            #   applied the change of basis to our qubits
            # next two lines are just setting up objects (not important)
            density_mat = qcp_program.Density(state=qbit)
            measure = qcp_program.Measure(density=density_mat)
            # now, we projectively measure and list the probabilities of
            #   each state in the computational basis
            projective_probs = measure.list_probs()
            # now we just randomly select on of the two states based on the
            #   probability distributino of projective_probs
            measurement = rm.choices(range(2), weights=projective_probs)[0]
            self.b.append(measurement) # add the measurement to self.b

    # used for removing self.y and self.b elements that don't align with Alice
    def prune(self, l):
        for element in l:
            del self.y[element]
            del self.b[element]
    
    # decodes Alice's message and converts to a string
    def decode(self, code):
        #short_key grabs a chunk of the key the size of the code
        short_key = [self.b.pop(0) for _ in range(len(code))]
        final_key = ''.join(map(str, short_key)) # convert to bistring
        # XOR the code from Alice and our key:
        bin_decoding = bin(int(code,2) ^ int(final_key, 2))[2:].zfill(len(code))
        return bits_to_m(bin_decoding) # convert back to a string
    
    # Bob's method for encoding his message
    def answer(self, message):
        bits = m_to_bits(message) # converst message to bits
        # answer_key grabs an adequate portion of the key
        answer_key = [self.b.pop(0) for _ in range(len(bits))]
        final_key = ''.join(map(str, answer_key)) # converts back to string
        if len(bits) != len(final_key):
            print('Length Issue!')
        # cypher XOR's the message and key
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        #   two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        cypher = bin(int(bits,2) ^ int(final_key,2))[2:].zfill(len(bits))
        return cypher # this is a bitstring

# this class creates object of type Eve, which tries to interept communication,
#   decrypt messages, send false messages, and is distributes the quantum key
#   for the BBM92 protocol
class EVE:

    def __init__(self):
        self.qk = [] # quantum key for BBM92 protocol
        self.fqk = [] # fake quantum key to send to Alice or Bob
        self.z = [] # list of bases to use to measure a quantum key
        self.c = [] # list of results of the quantum key measurement
        self.fz = [] # list of bases used in fake quantum key creation
        self.fc = [] # list of choice of qubit within given basis (for fake QK)
                     #for example: |0>,|+> are 0, while |1>,|-> are 1
    
    # creates quantum_key for BBm92 protocol
    def generate_key(self, size):
        self.qk = []
        phi = qcp_program.Qubit.q0().__matmul__(qcp_program.Qubit.q0()) + qcp_program.Qubit.q1().__matmul__(qcp_program.Qubit.q1())
        # phi is a bell state (normalized |00> + |11>)
        for i in range(size):
            self.qk.append(phi) # list of just bell states
    
    # creates the basis for measuring an intercepted quantum key
    def create_z(self, l, protocol):
        self.z = []
        if protocol == 1: # single-state (SS)
            for i in range(l):
                r_num = rm.randint(0,2)
                self.z.append(r_num)
        elif protocol == 0 or protocol == 2 or protocol == 3:
            for i in range(l):
                r_num = rm.randint(0,1)
                self.z.append(r_num)
        else: raise ValueError("protocol unexpected")

    # measures an intercepted quantum key and adds results to self.c
    def measure_key(self, key):
        if len(self.z) != len(key):
            print("z length "+ str(len(self.z)))
            print("key length " + str(len(key)))
            raise ValueError("Key and measurement device length mismatch.")
        for i in range(len(self.z)):
            qbit = key[i]
            if self.z[i] == 0: pass # 0 means computational basis
            elif self.z[i] == 1: # 1 means hadamard basis
                H = qcp_program.Gate.Hadamard()
                qbit = H.__mul__(qbit)
            elif self.z[i] == 2: # 2 means third basis for SS
                H = qcp_program.Gate.Hadamard()
                P = qcp_program.Gate.P_Gate(-np.pi/2)
                qbit = H.__mul__(P.__mul__(qbit))
            # we can now measure in computational basis as we have already
            #   applied the change of basis to our qubits
            # next two lines are just setting up objects (not important)
            density_mat = qcp_program.Density(state=qbit)
            measure = qcp_program.Measure(density=density_mat)
            # now, we projectively measure and list the probabilities of
            #   each state in the computational basis
            projective_probs = measure.list_probs()
            # now we just randomly select on of the two states based on the
            #   probability distributino of projective_probs
            measurement = rm.choices(range(2), weights=projective_probs)[0]
            self.c.append(measurement)
    
    # this method allows Alice and Bob to measure the Eve's bell states
    def others_measure(self, x, y):
        if len(x) != len(y): raise ValueError('measurement lengths mismatch')
        a_l = [] # list of measurement results for Alice
        b_l = [] # list of measurement results for Bob
        for i in range(len(x)):
            # these next four lines convert the 0's and 1's of x and y
            #   (being the basis) choices, to actual matrices
            if x[i] == 0: m1 = qcp_program.Identity # comp. basis => identity
            elif x[i] == 1: m1 = qcp_program.Hadamard # Hadamadrd basis
            if y[i] == 0: m2 = qcp_program.Identity
            elif y[i] == 1: m2 = qcp_program.Hadamard
            # big_gate is a tensor product of alice and bob's measurement
            #   chocies
            big_gate = m1.__matmul__(m2)
            # now, we transform bell states into this basis
            list = big_gate.__mul__(self.qk[i])
            list_v = list.vector # find the vector of this state
            # multiply amplitude vector by complex conjugate and normalize:
            probs = [i*np.conj(i) for i in list_v]/sum([i*np.conj(i) for i in list_v])
            p = np.array(probs,dtype=float) # converts to array of floats
            # now, we measure (choose an index according to probability)
            #   distribution of state
            index = np.random.choice(len(p), p=p)
            if index == 0: # this would be |00>
                a_l.append(0)
                b_l.append(0)
            elif index == 1: # this would be |10>
                a_l.append(1)
                b_l.append(0)
            elif index == 2: # this would be |01>
                a_l.append(0)
                b_l.append(1)
            elif index == 3: # this would be |11>
                a_l.append(1)
                b_l.append(1)
        return a_l, b_l
    
    # this allows Eve to create a fake quantum key based on the protocol
    def generate_fake_key(self, protocol, key_length):
        self.fqk = []
        if protocol == 0 or protocol == 3: # BB84 or BBM92
            for i in range(key_length):
                r_num = rm.randint(0,3)
                if r_num == 0:
                    self.fqk.append(qcp_program.Qubit.q0())
                    self.fz.append(0)
                    self.fc.append(0)
                elif r_num == 1:
                    self.fqk.append(qcp_program.Qubit.q1())
                    self.fz.append(0)
                    self.fc.append(1)
                elif r_num == 2:
                    self.fqk.append(qcp_program.Qubit.qp())
                    self.fz.append(1)
                    self.fc.append(0)
                elif r_num == 3:
                    self.fqk.append(qcp_program.Qubit.qm())
                    self.fz.append(1)
                    self.fc.append(1)
        elif protocol == 1: # six-state (SS)
            for i in range(key_length):
                r_num = rm.randint(0,5)
                if r_num == 0:
                    self.fqk.append(qcp_program.Qubit.q0())
                    self.fz.append(0)
                    self.fc.append(0)
                elif r_num == 1:
                    self.fqk.append(qcp_program.Qubit.q1())
                    self.fz.append(0)
                    self.fc.append(1)
                elif r_num == 2:
                    self.fqk.append(qcp_program.Qubit.qp())
                    self.fz.append(1)
                    self.fc.append(0)
                elif r_num == 3:
                    self.fqk.append(qcp_program.Qubit.qm())
                    self.fz.append(1)
                    self.fc.append(1)
                elif r_num == 4:
                    self.fqk.append(qcp_program.Qubit.qpi())
                    self.fz.append(2)
                    self.fc.append(0)
                elif r_num == 5:
                    self.fqk.append(qcp_program.Qubit.qmi())
                    self.fz.append(2)
                    self.fc.append(1)
        elif protocol == 2: # B92 QKD
            for i in range(key_length):
                r_num = rm.randint(0,1)
                if r_num == 0:
                    self.fqk.append(qcp_program.Qubit.q0())
                    self.fc.append(0)
                elif r_num == 1:
                    self.fqk.append(qcp_program.Qubit.qm())
                    self.fc.append(1)
        else: raise ValueError("We aren't using any of our protocols")

    # used for removing self.z,self.c,self.fc,self.fz elements that don't align
    #   with Alice and Bob
    def prune(self, l):
        for element in l:
            del self.z[element]
            del self.c[element]
            del self.fc[element]
            # as B92 isn't initated with a basis choice:
            if protocol != 2: del self.fz[element]
            
    # enables Eve to decode intercepted messages from Eve
    #   won't explain specifics as have many decode methods above
    def decode(self, recieved_message):
        l = len(recieved_message)
        short_key = [self.c.pop(0) for _ in range(l)]
        final_key = ''.join(map(str, short_key))
        bit_m = bin(int(recieved_message,2) ^ int(final_key,2))[2:].zfill(len(recieved_message))
        return bits_to_m(bit_m)
    
    # enables Eve to decode intercepted messages from Bob
    #   those messages were encoded with Eve's fake key
    #   won't explain specifics as have many decode methods above
    def fdecode(self, recieved_message):
        l = len(recieved_message)
        short_key = [self.fc.pop(0) for _ in range(l)]
        final_key = ''.join(map(str, short_key))
        bit_m = bin(int(recieved_message,2) ^ int(final_key,2))[2:].zfill(len(recieved_message))
        return bits_to_m(bit_m)
          


game_mode=None # 0 is virtual, 1 is multiplayer
protocol=None # 0 is BB84, 1 is SS, 2 is B92, 3 is BBM92
single_player = None # if virtual, single_player is 'BOB' or 'ALICE'
Alice = None # Alice object
Bob = None # Bob object
alice_message = None
bob_message = None
key_length = None
decoded_alice_message = ''
decoded_bob_message = ''
error_rate = None # found rate of errors in shared key segment
eavesdropper = False
Eve = EVE()

# method that allows Alice and Bob to compare a fracton of their key to
#   check for errors
# returns a boolean, being whether the public check passed or failed
def public_compare():
    global error_rate, error_tolerance, protocol
    if fraction.get() == 0: return True # if no bits are checked, we pass
    else:
        n = len(Alice.a)*fraction.get() # number of bits to check
        count = 0
        for _ in range(int(np.ceil(n))):
            i = rm.randint(0,len(Alice.a)-1) # grabs a random bit
            if Alice.a[i] == Bob.b[i]: pass # if the bit is the same we're good
            else: count += 1 # if the bit is different, add a counter
            del Alice.a[i]
            if protocol != 2: del Alice.x[i] # B92 doesn't start with bases
            del Bob.b[i]
            del Bob.y[i]
        error_rate = round(count/n, 2) # bad bits/total bits to 2 decimal places
        if error_rate >= error_tolerance.get(): return False
        else: return True

# method for cleaering a frame and opening a new one
#   also allows for global variable setting
def next_page(old_page, new_page, variable, value):
    global game_mode,protocol,single_player, eavesdropper
    old_page.pack_forget() # clears old_page
    new_page.pack(fill='both', expand=True) # creates new page
    if variable == "game_mode": game_mode = value
    elif variable == "protocol": protocol = value
    elif variable == 'single_player': single_player = value
    elif variable == 'eavesdropper': eavesdropper = value

root = tk.Tk() # begins GUI
root.title("Quantum Cryptography Simulator")
root.geometry('1500x750')

# global TK variables
fraction = tk.DoubleVar(value=0.0) # fraction of bits to be publicly checked
error_tolerance = tk.DoubleVar(value=0.0) # maximum errors we accept

# starting page
entry_frame = tk.Frame(root, background="deepskyblue2")
entry_frame.pack(fill="both", expand=True)

start_text = 'Welcome to our quantum cryptography simulator!'
start_header = tk.Label(entry_frame, text=start_text, font=("Arial", 40, "bold"))
start_header.pack(side='top', pady=10)

start_button = tk.Button(entry_frame, text="START", font=("Arial", 72), relief='flat', borderwidth=0, command=lambda: next_page(entry_frame,game_mode_frame, None, None))
start_button.pack(expand=True) #expand=True centers the button in the frame

# second page, allows choosing game mode 
game_mode_frame = tk.Frame(root, background='deepskyblue2')

game_mode_text = "PLEASE SELECT YOUR GAME MODE:"
game_mode_description = 'You can choose Virtual QCrypto to converse securely with a virtual bot or MultiPlayer QCrypto to converse securely with a second player.'

game_mode_header = tk.Label(game_mode_frame, text=game_mode_text, font=("Arial", 40, "bold"))
game_mode_header.pack(side='top', pady=10)
game_mode_subheader = tk.Label(game_mode_frame, text=game_mode_description, font=("Arial", 18),fg='grey')
game_mode_subheader.pack(side='top', pady=10)

virtual_game_mode_button = tk.Button(game_mode_frame, text='Virtual QCrypto', font=("Arial", 36), relief='flat', borderwidth=5, command=lambda: next_page(game_mode_frame, eve_frame, "game_mode", 0))
virtual_game_mode_button.pack(padx=0, expand=True)

multiplayer_game_mode_button = tk.Button(game_mode_frame, text='MultiPlayer QCrypto', font=("Arial", 36), relief='flat', borderwidth=5, command=lambda: next_page(game_mode_frame, eve_frame, "game_mode", 1))
multiplayer_game_mode_button.pack(padx=0, expand=True)

eve_frame = tk.Frame(root, background='deepskyblue2')

eve_text = 'Would you liked an EVE to attempt to disrupt your communication?'
eve_subtext = 'An EVE is an unwanted eavesdropper.'

eve_header = tk.Label(eve_frame, text=eve_text, font=("Arial", 32, "bold"))
eve_subheader = tk.Label(eve_frame, text=eve_subtext,font=("Arial", 18),fg='grey')
eve_header.pack(side='top', pady=10)
eve_subheader.pack(side='top', pady=10)

yeve_button = tk.Button(eve_frame, text='YES', font=("Arial", 24), relief='flat', command=lambda: next_page(eve_frame, protocol_frame, "eavesdropper", True))
neve_button = tk.Button(eve_frame, text='NO', font=("Arial", 24), relief='flat', command=lambda: next_page(eve_frame, protocol_frame, "eavesdropper", False))
yeve_button.pack(side='top', pady=10)
neve_button.pack(side='top', pady=10)

protocol_frame = tk.Frame(root, background='deepskyblue2')

protocols_text = "PLEASE SELECT YOUR QKD PROTOCOL:"
protocols_description = "You may choose to communicate via BB84, SS, B92, or BBM92 protocols."

protocols_header = tk.Label(protocol_frame, text=protocols_text, font=("Arial", 40, "bold"))
protocols_header.pack(side='top', pady=20)
protocols_subheader = tk.Label(protocol_frame, text=protocols_description, font=("Arial", 18),fg='grey')
protocols_subheader.pack(side='top', pady=20)


def protocol_choice(old_frame,protocol_num):
    global game_mode,protocol
    if game_mode == 0:
        # if playing virtual, must choose your character
        next_page(old_frame, virtual_character_frame, "protocol", protocol_num)
    elif game_mode == 1:
        next_page(old_frame, values_frame, "protocol", protocol_num)
    

bb84_button = tk.Button(protocol_frame, text='BB84', font=("Arial", 24), relief='flat', command=lambda: protocol_choice(protocol_frame, 0))
bb84_button.pack(side='top',pady=10)

ss_button = tk.Button(protocol_frame, text='SS', font=("Arial", 24), relief='flat', command=lambda: protocol_choice(protocol_frame, 1))
ss_button.pack(side='top',pady=10)

b92_button = tk.Button(protocol_frame, text='B92', font=("Arial", 24), relief='flat', command=lambda: protocol_choice(protocol_frame, 2))
b92_button.pack(side='top',pady=10)

bbm92_button = tk.Button(protocol_frame, text='BBM92', font=("Arial", 24), relief='flat', command=lambda: protocol_choice(protocol_frame, 3))
bbm92_button.pack(side='top',pady=10)

virtual_character_frame = tk.Frame(root, background='deepskyblue2')

virtual_characters_text = "PLEASE SELECT TO PLAY AS EITHER ALICE OR BOB"
virtual_characters_header = tk.Label(virtual_character_frame, text=virtual_characters_text, font=("Arial", 40, "bold"))
virtual_characters_header.pack(side='top', pady=20)

alice_button = tk.Button(virtual_character_frame, text='ALICE', font=("Arial", 24), relief='flat', command=lambda: next_page(virtual_character_frame, values_frame, "single_player", "ALICE"))
alice_button.pack(side='top',pady=20)
bob_button = tk.Button(virtual_character_frame, text='BOB', font=("Arial", 24), relief='flat', command=lambda: next_page(virtual_character_frame, values_frame, "single_player", "BOB"))
bob_button.pack(side='top',pady=20)

values_frame = tk.Frame(root, background='deepskyblue2')

values_text = "PARAMETERS"
values_description = "Please select both a fraction and QBER threshold."

values_header = tk.Label(values_frame, text=values_text, font=("Arial", 40, "bold"))
values_header.pack(side='top', pady=20)
values_subheader = tk.Label(values_frame, text=values_description, font=("Arial", 18),fg='grey')
values_subheader.pack(side='top', pady=20)

# methods displays the slider value for error checking fraction
def display_slider_fraction(x):
    fraction.set(value=float(x)/100)
    fraction_slider_label.config(text=f"Fraction for Checking: {fraction.get()*100}%")

# method displays the slider value for error tolerance
def display_slider_error_tolerance(x):
    error_tolerance.set(value=float(x)/100)
    error_tolerance_slider_label.config(text=f"Error Threshold: {error_tolerance.get()*100}%")

fraction_slider_label = tk.Label(values_frame, text="Fraction for Checking: 0%", font=("Arial", 24))
error_tolerance_slider_label = tk.Label(values_frame, text="Error Threshold: 0%", font=("Arial", 24))

fraction_slider = tk.Scale(values_frame, from_=0, to=100, orient='horizontal', font=("Arial", 14), length=200, command=display_slider_fraction)
error_tolerance_slider = tk.Scale(values_frame, from_=0, to=100, orient='horizontal', font=("Arial", 14), length=200, command=display_slider_error_tolerance)

fraction_slider_label.pack()
fraction_slider.pack()
error_tolerance_slider_label.pack()
error_tolerance_slider.pack()

# this method takes a current page, and moves us to a new page where our first
#   communicator inputs their message
def first_messenger(page):
    global game_mode, single_player, Alice, Bob, fraction, error_tolerance, alice_message, key_length
    Bob = BOB()
    if protocol == 3:
        key_length = 2000 # arbitrary (large) choice
        Eve.generate_key(key_length)
    if game_mode == 1 or single_player  == 'ALICE':
        next_page(page, alice_page, None, None) # go straight to Alice's page
    elif single_player == 'BOB':
        # randomly choose alice's message from the following list:
        alice_message = rm.choice(['Are all quantum states seperable?', 'How are you?', 'Did you know a group of ravens is called a treachery?'])
        Alice = ALICE(alice_message,fraction,error_tolerance)
        if protocol < 3:
            key_length = len(m_to_bits(alice_message))*10
            Alice.generate_big_key(protocol, key_length)
            Bob.create_y(key_length, protocol)
            if eavesdropper:
                Eve.create_z(key_length, protocol)
                Eve.measure_key(Alice.quantum_key)
                Eve.generate_fake_key(protocol,key_length)
                Bob.measure_key(Eve.fqk)
            else: Bob.measure_key(Alice.quantum_key)
        elif protocol == 3:
            Alice.create_x(key_length)
            Bob.create_y(key_length, protocol)
            if eavesdropper:
                Eve.create_z(key_length, protocol)
                Alice.a, Eve.c = Eve.others_measure(Alice.x, Eve.z)
                Eve.generate_fake_key(protocol,key_length)
                Bob.measure_key(Eve.fqk)
            else: Alice.a, Bob.b = Eve.others_measure(Alice.x, Bob.y)
        if len(Alice.a) != len(Bob.b): raise ValueError("Bob and Alice length mismatch.")
        bad_indices = []
        if protocol == 0 or protocol == 1 or protocol == 3:
            for i in range(len(Alice.x)):
                if Alice.x[i] != Bob.y[i]: bad_indices.insert(0,i)
        elif protocol == 2:
            for i in range(len(Bob.y)):
                if Bob.y[i] == Bob.b[i]: bad_indices.insert(0,i)
        Alice.prune(bad_indices)
        Bob.prune(bad_indices)
        if eavesdropper: Eve.prune(bad_indices)
        if public_compare():
            cyphertext = Alice.encode()
            plaintext = str(Bob.decode(cyphertext))
            bob_recieved.config(text="ALICE's message was: " + plaintext)
            if eavesdropper:
                stolen = str(Eve.decode(cyphertext))
                stolen_text.config(text="EVE discretily gained some information about ALICE's message and decoded: " + stolen)
            next_page(page, bob_page, None, None)
        else:
            error_header.config(text='The found error rate of ' + str(error_rate) + ' exceeded our error tolerance of ' + str(error_tolerance.get()))
            next_page(page, error_page, None, None)

first_continue_button = tk.Button(values_frame, text='CONTINUE', font=("Arial", 24, "bold"), relief='flat', command=lambda: first_messenger(values_frame))
first_continue_button.pack(pady=30)

alice_page = tk.Frame(root, background='deepskyblue2')
bob_page = tk.Frame(root, background='deepskyblue2')

error_page = tk.Frame(root, background='deepskyblue2')

error_text = ''
error_subtext = 'This indicates there may have been an eavesdropper. Thus, we aborted the communication.'

error_header = tk.Label(error_page, text=error_text, font=("Arial", 24, "bold"))
error_header.pack(side='top', pady='20')
error_subheader = tk.Label(error_page, text=error_subtext, font=("Arial", 18), fg='grey')
error_subheader.pack(side='top', pady='20')

error_restart_button = tk.Button(error_page, text='START OVER', font=("Arial", 24), relief='flat', command=lambda: next_page(error_page, entry_frame, None, None))
error_restart_button.pack(pady=30)

alice_text = "Welcome to ALICE's Secure Communication Portal"
alice_description = "Please share your message for BOB below:"

alice_header = tk.Label(alice_page, text=alice_text, font=("Arial", 40, "bold"))
alice_header.pack(side='top', pady='20')
alice_subheader = tk.Label(alice_page, text=alice_description, font=("Arial", 18), fg='grey')
alice_subheader.pack(side='top', pady='20')

alice_box = tk.Text(alice_page, height=20, width=40)
alice_box.pack(pady=20)

# method allows saving of Alice's text and progressing to next frame
def alice_save_text():
    global alice_message, single_player, key_length, fraction, error_tolerance, decoded_alice_message, protocol, Alice, Bob
    alice_message = alice_box.get("1.0", tk.END).strip() # strips text from box
    Alice = ALICE(alice_message, fraction, error_tolerance)
    if protocol < 3:
        key_length = len(m_to_bits(alice_message))*10
        Alice.generate_big_key(protocol, key_length)
        Bob.create_y(key_length, protocol)
        if eavesdropper:
            Eve.create_z(key_length, protocol)
            Eve.measure_key(Alice.quantum_key)
            Eve.generate_fake_key(protocol,key_length)
            Bob.measure_key(Eve.fqk)
        else: Bob.measure_key(Alice.quantum_key)
    elif protocol == 3:
        Alice.create_x(key_length)
        Bob.create_y(key_length, protocol)
        if eavesdropper:
            Eve.create_z(key_length, protocol)
            Alice.a, Eve.c = Eve.others_measure(Alice.x, Eve.z)
            Eve.generate_fake_key(protocol,key_length)
            Bob.measure_key(Eve.fqk)
        else: Alice.a, Bob.b = Eve.others_measure(Alice.x, Bob.y)
    if len(Alice.a) != len(Bob.b): raise ValueError("Bob and Alice length mismatch.")
    bad_indices = []
    if protocol == 0 or protocol == 1 or protocol == 3:
        for i in range(len(Alice.x)):
            if Alice.x[i] != Bob.y[i]: bad_indices.insert(0,i)
    elif protocol == 2:
        for i in range(len(Bob.y)):
            if Bob.y[i] == Bob.b[i]: bad_indices.insert(0,i)
    Alice.prune(bad_indices)
    Bob.prune(bad_indices)
    if eavesdropper: Eve.prune(bad_indices)
    if public_compare():
        code = Alice.encode()
        decoded_alice_message = str(Bob.decode(code))
        bob_recieved.config(text='ALICEs message was: ' + decoded_alice_message)
        if eavesdropper:
            stolen = str(Eve.decode(code))
            stolen_text.config(text="EVE discretily gained some information about ALICE's message and decoded: " + stolen)
        if game_mode == 0:
            bob_message = rm.choice(['Yes', 'No', 'Maybe', 'Sure'])
            cypher = Bob.answer(bob_message)
            decoded_bob_message = str(Alice.decode(cypher))
            if eavesdropper:
                stolen2 = str(Eve.decode(cypher))
                stolen_text2.config(text="EVE discretily gained some information about BOB's message and decoded: " + stolen2)
                restart_game_button.config(text='Click to Continue Communicating')
            restart_page_subheader.config(text=decoded_bob_message)
            next_page(alice_page, restart_page, None,None)
        elif game_mode == 1:
            next_page(alice_page, bob_page, None, None)
    else:
        error_header.config(text='The found error rate of ' + str(error_rate) + ' exceeded our error tolerance of ' + str(error_tolerance.get()))
        next_page(alice_page, error_page, None, None)

alice_send = tk.Button(alice_page, text='SEND',font=("Arial", 24), relief='flat', command=alice_save_text)
alice_send.pack(pady=20)

bob_text = "Welcome to BOB's Secure Communication Portal"
bob_d1 = 'ALICEs message was: '
bob_description = "Please share your message for ALICE below:"

bob_header = tk.Label(bob_page, text=bob_text, font=("Arial", 40, "bold"))
bob_header.pack(side='top', pady='20')
bob_recieved = tk.Label(bob_page, text=bob_d1, font=("Arial", 32))
bob_recieved.pack(side='top', pady='10')
stolen_text = tk.Label(bob_page, text='You may now respond', font=("Arial", 32))
stolen_text.pack(side='top', pady='10')
bob_subheader = tk.Label(bob_page, text=bob_description, font=("Arial", 18), fg='grey')
bob_subheader.pack(side='top', pady='20')

bob_box = tk.Text(bob_page, height=20, width=40)
bob_box.pack(pady=20)

# allows saving of Bob text and progressing onto next frame
def bob_save_text():
    global bob_message, single_player, decoded_bob_message, Alice, Bob
    bob_message = bob_box.get("1.0", tk.END).strip()
    cypher = Bob.answer(bob_message)
    decoded_bob_message = str(Alice.decode(cypher))
    if eavesdropper:
        textB =  str(Eve.decode(cypher))
        stolen_text2.config(text="EVE discretily gained some information about ALICE's message and decoded: " + textB)
        restart_game_button.config(text='Click to Continue Communicating')
    restart_page_subheader.config(text=decoded_bob_message)
    next_page(bob_page,restart_page, None, None)

bob_send = tk.Button(bob_page, text='SEND', font=("Arial", 24), relief='flat', command=bob_save_text)
bob_send.pack(pady=20)

# allows game to continue indefinetily by moving back to the first characters
#   messaging page
def restart(page):
    first_messenger(page)

restart_page = tk.Frame(root, background='deepskyblue2')

restart_page_text = "ALICE, BOB's response has been recieved and decoded as:"
restart_page_description = decoded_bob_message

restart_page_header = tk.Label(restart_page, text=restart_page_text, font=("Arial", 40, "bold"))
restart_page_header.pack(side='top', pady='20')

restart_page_subheader = tk.Label(restart_page, text=restart_page_description, font=("Arial", 24), fg='grey')
restart_page_subheader.pack(side='top', pady='20')

stolen_text2 = tk.Label(restart_page, text='Click Below to Continue Communicating', font=("Arial", 24), fg='grey')
stolen_text2.pack(side='top', pady='20')

restart_game_button = tk.Button(restart_page, text='Continue Communicating',font=("Arial", 24), relief='flat', command=lambda: restart(restart_page))
restart_game_button.pack(pady=20)

root.mainloop() # closes the GUI loop
print("Script completed run.")