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
import qcp_program


# I was thinking of making a little game of BB84 where you play different
#   characters to both test and understand the scheme.
# We could pull from the ChatGPT API to create fully encrypted automated
#   conversations.
# We could also just have it be a back and forth between two players
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
# ALICE:
#   Bob randomly comes up with a message of bitlength n that can be answered
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

# FOR NOW WE WILL FORGET ABOUT THIS


# SECOND: WE ARE GOING TO MAKE A SIMPLE BOB-CENTERED IMPLEMENTATION
#   okay we want Bob to be able to read-in messages randomly generate a key

def m_to_bits(m):
    return ''.join(format(ord(c), '08b') for c in m)

def bits_to_m(bit_m):
    if len(bit_m) % 8 != 0:
        # print('bit message is:  ' + bit_m)
        # print(len(bit_m) % 8)
        raise ValueError("Input bit string length must be a multiple of 8")
    chars = [bit_m[i:i+8] for i in range(0, len(bit_m), 8)]
    # print('the amount of chars is: ' + str(len(chars)))
    return ''.join(chr(int(char, 2)) for char in chars)

class BOB:
    def __init__(self, message, fraction, error_tolerance):
        self.message = message
        self.fraction = fraction
        self.error_tolerance = error_tolerance
        self.quantum_key = []
        self.x = []
        self.a = []
    
    def generate_big_key(self, protocol):
        big_key_length = (len(m_to_bits(self.message)) + 1)*4
        # I haven't implemented fraction here yet, just using *4 for safety
        if protocol.lower() == 'bb84':
            for i in range(big_key_length):
                r_num = rm.randint(0,3)
                if r_num == 0:
                    self.quantum_key.append(qcp_program.Qubit.q0())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(0)
                    self.a.append(0)
                elif r_num == 1:
                    self.quantum_key.append(qcp_program.Qubit.q1())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(0)
                    self.a.append(1)
                elif r_num == 2:
                    self.quantum_key.append(qcp_program.Qubit.qp())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(1)
                    self.a.append(0)
                elif r_num == 3:
                    self.quantum_key.append(qcp_program.Qubit.qm())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(1)
                    self.a.append(1)
        elif protocol.lower() == 'ss':
            for i in range(big_key_length):
                r_num = rm.randint(0,5)
                if r_num == 0:
                    self.quantum_key.append(qcp_program.Qubit.q0())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(0)
                    self.a.append(0)
                elif r_num == 1:
                    self.quantum_key.append(qcp_program.Qubit.q1())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(0)
                    self.a.append(1)
                elif r_num == 2:
                    self.quantum_key.append(qcp_program.Qubit.qp())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(1)
                    self.a.append(0)
                elif r_num == 3:
                    self.quantum_key.append(qcp_program.Qubit.qm())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(1)
                    self.a.append(1)
                elif r_num == 4:
                    self.quantum_key.append(qcp_program.Qubit.qyp())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(2)
                    self.a.append(0)
                elif r_num == 5:
                    self.quantum_key.append(qcp_program.Qubit.qym())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.x.append(2)
                    self.a.append(1)
        elif protocol.lower() == 'b92':
            for i in range(big_key_length):
                r_num = rm.randint(0,1)
                if r_num == 0:
                    self.quantum_key.append(qcp_program.Qubit.q0())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.a.append(0)
                elif r_num == 1:
                    self.quantum_key.append(qcp_program.Qubit.qm())
                    # qcp_program.Qubit.q0().qubit_info()
                    self.a.append(1)
        else: raise ValueError("We aren't using any of our protocols")

                
    
    def prune(self, l):
        for element in l:
            del self.x[element]
            del self.a[element]

    def encode(self):
        bit_m = m_to_bits(self.message)
        short_key = [self.a.pop(0) for _ in range(len(bit_m))]
        # print('length of the message in bits is' + str(len(bit_m)))
        if len(bit_m) != len(short_key):
            raise ValueError("message bits and key are of different length")
        final_key = ''.join(map(str, short_key))
        # print('the OG key that bob uses is :' + str(final_key))
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        # two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        cypher = bin(int(bit_m,2) ^ int(final_key,2))[2:].zfill(len(bit_m))
        # print('the XORed message seems to be: ' + cypher)
        return cypher
    
    def decode(self, recieved_message):
        l = len(recieved_message)
        short_key = [self.a.pop(0) for _ in range(l)]
        final_key = ''.join(map(str, short_key))
        # print('Bob decode final key: ' + final_key)
        bit_m = bin(int(recieved_message,2) ^ int(final_key,2))[2:].zfill(len(recieved_message))
        return bits_to_m(bit_m)


class ALICE:
    def __init__(self):
        # self.response = None
        self.y = []
        self.b = []

    def create_y(self, l, protocol):
        if protocol.lower() == 'bb84' or protocol.lower() == 'b92' or protocol.lower() == 'bbm92':
            for i in range(l):
                r_num = rm.randint(0,1)
                self.y.append(r_num)
        elif protocol.lower() == 'ss':
            for i in range(l):
                r_num = rm.randint(0,1,2)
                self.y.append(r_num)


    def measure_key(self, key):
        if len(self.y) != len(key):
            print("y length"+ str(len(self.y)))
            print("key length" + str(len(key)))
            raise ValueError("Key and measurement device length mismatch.")
        for i in range(len(self.y)):
            qbit = key[i]
            if self.y[i] == 0: pass
            elif self.y[i] == 1:
                H = qcp_program.Gate.Hadamard()
                qbit = H.__mul__(qbit)
            elif self.y[i] == 2:
                H = qcp_program.Gate.Hadamard()
                P = qcp_program.Gate.P_Gate()
                qbit = H.__mul__(P.__mul__(qbit))
            # qbit.qubit_info()
            # null_gate = qcp_program.Gate.Identity()
            density_mat = qcp_program.Density(state=qbit)
            measure = qcp_program.Measure(density=density_mat)
            projective_probs = measure.list_proj_probs()
            measurement = rm.choices(range(2), weights=projective_probs)[0]
            self.b.append(measurement)
                
                


            # this should work for computational basis butif 

            # self.b.append(qcp_program.Measure()key[i].measure(self.y[i]))
            # so the goal here is I have a state in key[i] and a basis in the
            # y[i] and want to be able to measure it in that basis
    def prune(self, l):
        for element in l:
            del self.y[element]
            del self.b[element]
    
    def decode(self, code):
        short_key = [self.b.pop(0) for _ in range(len(code))]
        final_key = ''.join(map(str, short_key))
        # print('the OG key alice is about to use is: ' + str(final_key))
        bin_decoding = bin(int(code,2) ^ int(final_key, 2))[2:].zfill(len(code))
        return bits_to_m(bin_decoding)
    
    def answer(self, message):
        bits = m_to_bits(message)
        answer_key = [self.b.pop(0) for _ in range(len(bits))]
        final_key = ''.join(map(str, answer_key))
        # print('Alice final key: ' + final_key)
        if len(bits) != len(final_key):
            print('Length Issue!')
        # print('the key that bob uses is :' + str(final_key))
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        # two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        cypher = bin(int(bits,2) ^ int(final_key,2))[2:].zfill(len(bits))
        return cypher

class EVE:
    def __init__(self):
        self.z = []
    def generate_key(self, size):
        for i in range(size):
            self.z.append(qcp_program.Qubit.Phip()) # have to remember to implement this



played_before = False
current_protocol = None
game_mode = None
BPlayer = False
APlayer = False
EPlayer = False
fraction = None
error_tolerance = None

# what should my order be? 

# you start by agreeing on a crypto scheme

def play():
    global played_before
    global current_protocol
    global game_mode
    global BPlayer
    global APlayer
    global EPlayer
    global fraction
    global error_tolerance
    if not played_before:
        current_protocol = input('Please enter your choice of QKD protocol: ')
        while current_protocol.lower() != 'bb84' and current_protocol.lower() != 'ss' and current_protocol.lower() != 'b92' and current_protocol.lower() != 'bbm92':
            current_protocol = input('Error. Input must be one of [bb84,ss,b92,bbm92]. Please try again: ')
        game_mode = input('Please enter your game mode, VIRTUALqc or MULTIPARTYqcrypto: ')
        while game_mode.lower() != 'virtualqcrypto' and game_mode.lower() != 'multipartyqcrypto':
            game_mode = input('Error. Input must be one of [VIRTUALqcrypto, MULTIPARTYqcrypto]. Please try again: ')
        if game_mode.lower() == 'virtualqcrypto':
            character_choice = input('Welcome to our quantum cryptography simulator! Here you get ' + \
                                'the chance to play one of three characters: ' + \
                                'BOB, ALICE, or EVE. BOB gets to ask questions ' + \
                                'and prompt a vitual ALICE, engaging in a back ' + \
                                'and forth discussion over a quantum-secure '+ \
                                'line of communication. ALICE gets to answer ' + \
                                'questions prompted by a virtual BOB over a ' + \
                                'quantum-secure line of commmunicataion. EVE ' + \
                                'discretily attempts to disrupt this ' + \
                                'communication. Please enter your choice of ' + \
                                'character: ')
            while character_choice.lower() != 'bob' and character_choice.lower() != 'alice' and character_choice.lower() != 'eve':
                character_choice = input('Error. Please choose one of [BOB, ALICE, EVE]: ')
            if character_choice.lower() == 'bob': BPlayer = True
            elif character_choice.lower() == 'alice': APlayer = True
            elif character_choice.lower() == 'eve': EPlayer = True
        elif game_mode.lower() == 'multipartyqcrypto':
            bob_value = input('If there is a BOB player please enter `Y`, else enter `N`: ')
            while bob_value.lower() != 'y' and bob_value.lower() != 'n':
                bob_value = input('Error. Please enter either Y or N: ')
            if bob_value.lower() == 'y': BPlayer = True
            elif bob_value.lower() == 'n': BPlayer = False
            alice_value = input('If there is a ALICE player please enter `Y`, else enter `N`: ')
            while alice_value.lower() != 'y' and alice_value.lower() != 'n':
                alice_value = input('Error. Please enter either Y or N: ')
            if alice_value.lower() == 'y': APlayer = True
            elif alice_value.lower() == 'n': APlayer = False
            eve_value = input('If there is a EVE player please enter `Y`, else enter `N`: ')
            while eve_value.lower() != 'y' and eve_value.lower() != 'n':
                eve_value = input('Error. Please enter either Y or N: ')
            if eve_value.lower() == 'y': EPlayer = True
            elif eve_value.lower() == 'n': EPlayer = False
        fraction = float(input("Thank you. Now, please input the fraction of bits you want to use to check for error: "))
        while fraction >= 1 or fraction < 0:
            fraction = float(input("Error checking fraction must be in [0,1), please try again: "))
        error_tolerance = float(input("Now, please describe your error tolerance: "))
        while error_tolerance >= 1 or error_tolerance < 0:
            error_tolerance = float(input("Error tolerance must be in [0,1), please try again: "))

    if BPlayer:
        bob_message = input('Hi BOB, thanks for playing. Please enter your ' + \
                           'secure question for ALICE: ')
    else: bob_message = 'Are all quantum states seperable?'

    Bob = BOB(bob_message,fraction,error_tolerance)
    Alice = ALICE()
    Eve = EVE()
    n = (len(m_to_bits(bob_message))+1)*4 # THIS SHOULD BE CHANGED
    if current_protocol.lower() == 'bb84' or current_protocol.lower() == 'ss' or current_protocol.lower() == 'b92':
        Bob.generate_big_key(current_protocol)
    elif current_protocol.lower() == 'bbm92': Eve.generate_big_key(n) # remember to implement n (size of key being generated)
    Alice.create_y(n)
    Alice.measure_key(Bob.quantum_key)
    if len(Bob.x) != len(Alice.y): raise ValueError("Bob and Alice length mismatch.")
    bad_indices = []
    if current_protocol == 'bb84' or current_protocol == 'ss' or current_protocol == 'bm92':
        for i in range(len(Bob.x)):
            if Bob.x[i] != Alice.y[i]: bad_indices.insert(0,i)
    elif current_protocol == 'b92':
        for i in range(len(Alice.y)):
            if Alice.y[i] == Alice.b[i]: bad_indices.insert(0,i)

    Bob.prune(bad_indices)
    Alice.prune(bad_indices)
    # print('bob has a message:' + Bob.message)
    # print('in binary this is:' + m_to_bits(Bob.message))
    # print('if we then decode we return to: ' + bits_to_m(m_to_bits(Bob.message)))
    code = Bob.encode()
    # print('Bobs encoded message is: ' + code)
    print("BOB's message has been succesfully encoded.")
    message_alice_recieves = Alice.decode(code)
    print("Alice decodes the message: " + message_alice_recieves)
    if APlayer:
        a_message = input('Hi Alice, Please enter you response:' )
    else:
        a_message = rm.choice(['Yes', 'No', 'Maybe'])
    alice_answer = Alice.answer(a_message)
    # print(alice_answer)
    print("BOB has received and decoded Alice's message. The message: " + Bob.decode(alice_answer))
    repeat = input('Would you like to communicate again? ')
    if repeat.lower() == 'yes' or repeat.lower() == 'ya' or repeat.lower() == 'yeah' or repeat.lower() == 'yea' or repeat.lower() == 'ok' or repeat.lower() == 'okay':
        played_before = True
        play()
    elif repeat.lower() == 'no' or repeat.lower() == 'nope':
        played_before = False
        print('Thanks for playing!')
    


play()

print("Script completed run.")

# okay what should I do next with this:
# options:
#   make UI more enjoyable
#   figure out how to make more than 2 person communication
#   implement it such that Alice and Bob are two characters that have a line of
#       communication


# ORDER
#   two people are playing, first one chooses Bob, second chooses Alice
#   bob sends a message first, alice gets to see it and then respond etc
#   in the beginning we arbitarily initiate a key size, and they take turns eating it up
#   at some point some user will be prompted that they only have x chars left to send
#   and then they will have to create a new key (once prompted)
#   thus, they have encrypted indefinite communication
#   also, I should make it so that they actually exchange key info to see if it
#   has been infiltrated
#   i could have more specific technical stuff here
#   I could also create an EVE that gets to try and clone a state but idk how
#   this would work. 

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



# CURRENT THOUGHTS: There are the usual things I can add, but I first must
# include other algorithms that I could implement
# bb83 or something, other angles



# Other QKD Algorithsm to implement:
# - SS protocol
# - B92
# - BBM92
# others? choosing angle between states?

# compute how good a given angle is? like QBER