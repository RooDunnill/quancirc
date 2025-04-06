from numpy import set_printoptions
import logging
import atexit
import time
import matplotlib.pyplot as plt
from functools import wraps

start = time.time() 
p_prec: int = 3                                      #printing precision in matrices
linewid: int = 160                                   #width of the lines in the terminal
eig_threshold: int = 1000       #Do not go below 2    will always use a dense matrix to compute the eigenvalues when dim of matrix is below this
sparse_threshold: int = 1000                         #will always be dense when dim of matrix is below this
dense_limit = 24000                                  #will always be sparse when dim of matrix exceeds this
sparse_matrix_threshold: float = 0.9                 #fraction of non-zeros for the matrix to be able to convert to sparse
sparse_array_threshold: float = 0.9                  #fraction of non-zeros for the array to be able to convert to sparse
name_limit = 50                                      #the character limit of qubits and gates
logging_level = logging.INFO                        #chooses the detail for logging, use DEBUG for everything, INFO to avoid degubbing logs and CRITICAL to turn off
set_printoptions(precision=p_prec, suppress=True, floatmode="fixed")
set_printoptions(linewidth=linewid)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def custom_format(record):
    if record.levelno == logging.INFO:
        return "%(message)s"
    else:
        return "[%(levelname)s] %(asctime)s - %(message)s"

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        self._style._fmt = custom_format(record)
        return super().format(record)

# Setup
logging.basicConfig(level=logging_level, format="%(message)s")
logging.getLogger().handlers[0].setFormatter(SimpleFormatter())

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"Called function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def log_all_methods(cls):
    for attr_name, attr in cls.__dict__.items():
        attr = getattr(cls, attr_name)
        if callable(attr):
            setattr(cls, attr_name, log_function_call(attr))
    return cls

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