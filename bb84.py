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
    
    def generate_big_key(self):
        big_key_length = (len(m_to_bits(self.message)) + 1)*3
        # I haven't implemented fraction here yet, just using *3 for safety
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
    
    def prune(self, l):
        for element in l:
            del self.x[element]
            del self.a[element]

    def encode(self):
        bit_m = m_to_bits(self.message)
        short_key = self.a[:len(bit_m)]
        # print('length of the message in bits is' + str(len(bit_m)))
        if len(bit_m) != len(short_key):
            raise ValueError("message bits and key are of different length")
        final_key = ''.join(map(str, short_key))
        # print('the key that bob uses is :' + str(final_key))
        # bin(x)[2:] returns the binary version of x and [2:] removes the first
        # two character "0b" which we don't want
        # zfill just adds enough 0's in front to equal length of self.message
        cypher = bin(int(bit_m,2) ^ int(final_key,2))[2:].zfill(len(bit_m))
        # print('the XORed message seems to be: ' + cypher)
        return cypher
    
    def decode(self, recieved_message):
        short_key = self.a[len(self.a) - 1]
        if bin(short_key ^ recieved_message)[2:] == '0':
            return "NO"
        elif bin(short_key ^ recieved_message)[2:] == '1':
            return "YES"
        else:
            print (bin(short_key ^ recieved_message)[2:])
            raise ValueError("Alice's message is neither 0 or 1 and thus can't be decrypted.")

class ALICE:
    def __init__(self):
        # self.response = None
        self.y = []
        self.b = []

    def create_y(self, l):
        for i in range(l):
            r_num = rm.randint(0,1)
            self.y.append(r_num)

    def measure_key(self, key):
        if len(self.y) != len(key):
            print("y length"+ str(len(self.y)))
            print("key length" + str(len(key)))
            raise ValueError("Key and measurement device length mismatch.")
        for i in range(len(self.y)):
            qbit = key[i]
            if self.y[i] == 1:
                H = qcp_program.Gate.Hadamard()
                qbit = H.__mul__(qbit)
            # qbit.qubit_info()
            # null_gate = qcp_program.Gate.Identity()
            density_mat = qcp_program.Density(qubit=qbit)
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
        short_key = self.b[:len(code)]
        final_key = ''.join(map(str, short_key))
        # print('the key alice is about to use is: ' + str(final_key))
        bin_decoding = bin(int(code,2) ^ int(final_key, 2))[2:].zfill(len(code))
        return bits_to_m(bin_decoding)
    
    def answer(self):
        return rm.randint(0,1)


played_before = False
character_choice = None

def play():
    global played_before
    global character_choice
    if not played_before:
        character_choice = input('Welcome to our BB84 Simulator! Here you get ' + \
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
    if character_choice.lower() == 'bob':
        user_input = input('Hi BOB, thanks for playing. Please enter your ' + \
                           'secure question for ALICE: ')
        fraction = float(input("Thank you. Now, please input the fraction of bits you want to use to check for error: "))
        while fraction >= 1 or fraction < 0:
            fraction = float(input("Error checking fraction must be in [0,1), please try again: "))
        error_tolerance = float(input("Now, please describe your error tolerance: "))
        while error_tolerance >= 1 or error_tolerance < 0:
            error_tolerance = float(input("Error tolerance must be in [0,1), please try again: "))
        Bob = BOB(user_input,fraction,error_tolerance)
        Bob.generate_big_key()
        Alice = ALICE()
        Alice.create_y((len(m_to_bits(user_input))+1)*3)
        Alice.measure_key(Bob.quantum_key)
        if len(Bob.x) != len(Alice.y): raise ValueError("Bob and Alice length mismatch.")
        bad_indices = []
        for i in range(len(Bob.x)):
            if Bob.x[i] != Alice.y[i]: bad_indices.insert(0,i)
        Bob.prune(bad_indices)
        Alice.prune(bad_indices)
        # print('bob has a message:' + Bob.message)
        # print('in binary this is:' + m_to_bits(Bob.message))
        # print('if we then decode we return to: ' + bits_to_m(m_to_bits(Bob.message)))
        code = Bob.encode()
        # print('Bobs encoded message is: ' + code)
        print("Your message has succesfully been encoded.")
        message_alice_recieves = Alice.decode(code)
        # print("Alice recieves the message:" + message_alice_recieves)
        alice_answer = Alice.answer()
        # print(alice_answer)
        print("We have received and decoded Alice's message. The message: " + Bob.decode(alice_answer))
        repeat = input('Would you like to ask ALICE  another question? ')
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