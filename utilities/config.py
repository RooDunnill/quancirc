from numpy import set_printoptions
import atexit
import time
import matplotlib.pyplot as plt
start = time.time() 
p_prec: int = 2
linewid: int = 100
set_printoptions(precision=p_prec, suppress=True, floatmode="fixed")
set_printoptions(linewidth=linewid)

def prog_end():    #made it to make the code at the end of the program a little neater
    """Runs at the end of the program to call the timer and plot only at the end"""
    stop = time.time()
    interval: float = stop - start
    print(f"Program ran for {interval:.3f} seconds")
    plt.show()                             #makes sure to plot graphs after the function
atexit.register(prog_end)