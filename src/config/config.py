from numpy import set_printoptions
import logging
from datetime import datetime
import atexit
import time
import matplotlib.pyplot as plt
from functools import wraps

start = time.time() 
p_prec: int = 2                                      #printing precision in matrices
linewid: int = 160                                   #width of the lines in the terminal
eig_threshold: int = 1000       #Do not go below 2    will always use a dense matrix to compute the eigenvalues when dim of matrix is below this
sparse_threshold: int = 1000                         #will always be dense when dim of matrix is below this
dense_limit: int = 24000                                  #will always be sparse when dim of matrix exceeds this
sparse_matrix_threshold: float = 0.9                 #fraction of non-zeros for the matrix to be able to convert to sparse
sparse_array_threshold: float = 0.9                  #fraction of non-zeros for the array to be able to convert to sparse
name_limit: int = 50                                      #the character limit of qubits and gates
logging_level = logging.INFO                        #chooses the detail for logging, use DEBUG for everything, INFO to avoid degubbing logs and CRITICAL to turn off
set_printoptions(precision=p_prec, suppress=True, floatmode="fixed")
set_printoptions(linewidth=linewid)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
now = datetime.now()

def custom_format(record):
    if record.levelno == logging.INFO:
        return "%(message)s"
    else:
        return "[%(levelname)s] %(asctime)s - %(message)s"

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        self._style._fmt = custom_format(record)
        return super().format(record)


logging.basicConfig(level=logging_level, format="%(message)s")
logging.getLogger().handlers[0].setFormatter(SimpleFormatter())

def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ == "__setattr__":
            logging.debug(f"Calling function: {func.__name__} setting attribute {args[1]}")
        else:
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
    logging.info(f"Start Time: {now.strftime("%Y-%m-%d %H:%M:%S")}")
    logging.info("\n")
    logging.info("#" * linewid)
    logging.info(f"Welcome to Quancirc, this program is mostly a collection of all of my current interests. Some of the highlights include:".center(linewid, '^'))
    logging.info(f"Quantum Circuit: Can simulate a quantum circuit with Clifford + T Gates".center(linewid, '-'))
    logging.info(f"Algorithm section, currently have a Grover's search algorithm that uses the simulator and utilises FWHT".center(linewid, '-'))
    logging.info(f"Cryptographic protocols such as BB84".center(linewid, '-'))
    logging.info(f"Currently working on symbolic qubits, to allow for symbolic calculations".center(linewid, '-'))
    logging.info(f"If you want examples of how to use, there are a few files within the src.examples folder that can give an overview of some of the functions".center(linewid, "-"))
    logging.info(f"I hope you enjoy!".center(linewid, '-'))
    logging.info(f"By Roo Dunnill".center(linewid, '-'))
    logging.info("\n")

def prog_end():    #made it to make the code at the end of the program a little neater
    """Runs at the end of the program to call the timer and plot only at the end"""
    stop = time.time()
    interval: float = stop - start
    logging.info("\n" + "#" * linewid)
    logging.info(f"Program ran for {interval:.3f} seconds")
    plt.show()                             #makes sure to plot graphs after the function
atexit.register(prog_end)
startup_printout()