import numpy as np
import math
import time

__all__ = ["rsa_key_gen"]

def Ext_Euler_alg(phi, e, verbose):   #used to generate the d value
    d = 1    #starts at 1
    while (e*d) % phi != 1:  #runs till a sufficient d value is found
        d += 1
        print("d value is: " + str(d), end='\r') if verbose else None
    return d

def e_generator(phi, e_input):
    while math.gcd(e_input, phi) != 1:
        e_input += 2   #usually takes no time, 3,5,7 are all prime which makes it easier
    return e_input

def rsa_key_gen(p, q, e_input, verbose=True):
    ti = time.perf_counter()     #used to find the time elapsed
    n = p*q                      #finds the n value by timesing two primes together
    phi = (p-1)*(q-1)            #a simplified way to find the phi value
    e = e_generator(phi, e_input)         #applies the e generation function        
    d = Ext_Euler_alg(phi, e, verbose)    #applies the d generation function with the newly found e value
    tf = time.perf_counter()  
    return n, phi, e, d
