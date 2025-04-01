from numpy import set_printoptions
import atexit
import time
import matplotlib.pyplot as plt

start = time.time() 
p_prec: int = 3
linewid: int = 160
eig_threshold: int = 1000 #Do not go below 2
sparse_threshold: int = 1000
sparse_matrix_threshold: float = 0.9
sparse_array_threshold: float = 0.9
name_limit = 50
set_printoptions(precision=p_prec, suppress=True, floatmode="fixed")
set_printoptions(linewidth=linewid)

def startup_printout():
    print("#" * linewid)
    print(f"Welcome to Quancirc, this program is mostly a collection of all of my current interests. Some of the highlights include:")
    print(f"Quantum Circuit: Can simulate a quantum circuit with Clifford + T Gates")
    print(f"Algorithm section, currently have a Grover's search algorithm that uses the simulator and utilises FWHT")
    print(f"Cryptographic protocols such as BB84")
    print(f"Currently working on symbolic qubits, to allow for symbolic calculations")
    print(f"I hope you enjoy!")
    print(f"By Roo Dunnill")
    print("\n")

def prog_end():    #made it to make the code at the end of the program a little neater
    """Runs at the end of the program to call the timer and plot only at the end"""
    stop = time.time()
    interval: float = stop - start
    print("#" * linewid)
    print(f"Program ran for {interval:.3f} seconds")
    plt.show()                             #makes sure to plot graphs after the function
atexit.register(prog_end)
startup_printout()