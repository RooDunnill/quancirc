from ..circuit.classes.lightweight_circuit.circuit_lw import *
from ..gen_utilities.timer import Timer
from .algorithm_utilities.algorithm_errors import GroverSearchError
from ..circuit.circuit_utilities.layout_funcs import top_probs
__all__ = ["grover_search"]


def optimal_iterations(oracle_values, n: int, ) -> tuple[float, int]:
    """Calculates the best number of iterations for a given number of qubits"""
    search_space: int = 2**n
    op_iter: float = (np.pi/4)*np.sqrt((search_space)/len(oracle_values)) -1/2        #the standard equation to find the optimal iterations
    return op_iter, search_space

def compute_n(oracle_values, n_cap: int, balanced_param: float, it) -> int:
    """Computes the optimal number of qubits and thus search space by finding a good iteration value based on the specific algorithm ran within"""
    if isinstance(n_cap, int):                #check on the input type of self.ncap
        print(f"Using up to {n_cap} Qubits to run the search")
        max_oracle = max(oracle_values)             #finds the largest one to find the min qubits
        n_qubit_min = 1
        while max_oracle > 2**n_qubit_min:             #when picking the qubits, we need enough to allow the search space to be bigger than all the oracle values
            n_qubit_min += 1
        if n_qubit_min > n_cap:
            raise QC_error(f"The search space needed for this search is larger than the qubit limit {n_cap}.")
        if it == None:          #if no given iteration then will find it
            print(f"No iteration value given, so will now calculate the optimal iterations")
            n_qubit_range = np.arange(n_qubit_min, n_cap + 1, dtype=int)
            if isinstance(balanced_param, int):            #then is will take that value over going down
                int_val_floor = 1
                int_val_round = 0
                print(f"Now computing n for the optimal iteration")
                for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                    op_iter = optimal_iterations(oracle_values, i)[0]
                    if op_iter >= 1:
                        int_dist_floor: float = op_iter - np.floor(op_iter)  #finds the float value
                        int_dist_round: float = 2*abs(int_dist_floor-0.5) 
                        print(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                        if int_dist_round > int_val_round:            #iterates through to find the smallest distance from an integer
                            n: int = i
                            int_val_round: float = int_dist_round
                return n
        else:
            n = n_qubit_min
            print(f"Running the given {it} iterations with the minimum number of qubits {n}")
            return n
    raise GroverSearchError(f"The qubit limit cannot be of {type(n_cap)}, expected type int")

def phase_oracle(qub: Qubit, oracle_values: list) -> Qubit:          #the Grover phase oracle
    """Computes the phase flip produced from the oracle values"""
    qub.state[oracle_values] *= -1                     #flips the values in the index of the oracle values
    return qub

def iterate_alg(oracle_values, n, mode, it) -> Qubit:
    """The main core algorithm of the program where the phase gates and Hadamard gates are applied"""
    it_count = 0
    timer = Timer()
    print(f"Running FWHT algorithm:") if mode == "fast" else print(f"Running algorithm:")
    if mode == "fast":               #this is the faster variant of Grovers utilising the FWHT
        print(f"Using FWHT to compute {n} x {n} Hadamard gate application")
        fwht_mode = True
    else:
        had = Hadamard                   
        print(f"Initialising {n} x {n} Hdamard")
        for i in range(n-1):    #creates the qubit and also the tensored hadamard for the given qubit size
            had **= Hadamard              #if not fast then tensors up a hadamard to the correct size
            print(f"\r{i+2} x {i+2} Hadamard created", end="")    #allows to clear line without writing a custom print function in print_array
        print(f"\r",end="")
        print(f"\rHadamard and Quantum State created, time to create was: {timer.elapsed()[0]:.4f}")
        fwht_mode = False
    grovers_circuit = Circuit_LW(q=n, verbose=False)

    print(f"\rApplying Initial Hadamard                                                                     ", end="")
    grovers_circuit.add_gate(Hadamard, fwht=fwht_mode)
    while it_count < int(it):   #this is where the bulk of the computation actually occurs and is where the algorithm is actually applied
        print(f"\rIteration {it + 1}:                                                                        ", end="")
        print(f"\rIteration {it_count + 1}: Applying phase oracle                                            ", end="")
        grovers_circuit.state = phase_oracle(grovers_circuit.state, oracle_values)              #STEP 2   phase flips the given oracle values
        print(f"\rIteration {it_count + 1}: Applying first Hadamard                                         ", end="")
        grovers_circuit.add_gate(Hadamard, fwht=fwht_mode)                                         #STEP 3 Applies the hadamard again
        print(f"\rIteration {it_count + 1}: Flipping the Qubits phase except first Qubit                     ", end="")
        grovers_circuit.state.state *= -1           #inverts all of the phases of the qubit values             STEP 4a
        grovers_circuit.state.state[0] *= -1              #inverts back the first qubits phase                 STEP 4b
        print(f"\rIteration {it_count + 1}: Applying second Hadamard                                         ", end="")
        grovers_circuit.add_gate(Hadamard, fwht=fwht_mode)
        it_count += 1                   #adds to the iteration counter
        print(f"\r                                                                                     Time elapsed:{timer.elapsed()[0]:.4f} secs", end="")
    print(f"\r",end="")
    print(f"\rFinal state calculated. Time to iterate algorithm: {timer.elapsed()[1]:.4f} secs                                                           ")
    print(f"Computing Probability Distribution of States")
    probs = grovers_circuit.list_probs()
    print(f"Finding the probabilities for the top n Probabilities (n is the number of oracle values)")
    sorted_arr = top_probs(probs, n=len(oracle_values))         #finds the n top probabilities
    return sorted_arr

def run(oracle_values, n, n_cap, it, mode, verbose, rand_ov, balanced_param):     #Grovers algorithm, can input the number of qubits and also a custom amount of iterations
    """This is the function to initiate the search and compiles the other functions together and prints the values out"""
    Grover_timer = Timer()
    print("=" * linewid)
    if rand_ov:
        print(f"Grovers search with random oracle values")
    else:
        print(f"Grovers search with oracle values: {oracle_values}")
    if n == None:               #if the number of qubits required is not given then run:
        n = compute_n(oracle_values, n_cap, balanced_param, it)
        search_space: int = 2**n       #computes the final search space for the chosen n
        print(f"Using {n} Qubits with a search space of {search_space} to get the best accuracy")
    elif isinstance(n, int):
        search_space: int = 2**n       #computes the search space for the n provided
        print(f"Using {n} Qubits with a search space of {search_space}")
    else:
        raise QC_error(f"self.n is of the wrong type {type(n)}, expected type int")

    op_iter = optimal_iterations(oracle_values, n)[0]
    if it == None:     #now picks an iteration value
        it = int(np.floor(op_iter))          #TODOO CURRENTLY THE ITERATION MODE IS NOT IMPLEMENTED
        
        if it < 1:    #obviously we cant have no iterations so it atleast does 1 iteration
            it = 1.0       #more of a failsafe, will almost certainly be wrong as the percentage diff from the it value to 1 will be large?
            print(f"Computing 1 iteration, most likely will not be very accurate")
        else:
            print(f"Optimal number of iterations are: {op_iter:.3f}")
            print(f"Will run {it} iterations")
    elif isinstance(it, int):
        print(f"Number of iterations to perform are: {it}")
    else:
        raise GroverSearchError(f"Iterations cannot be of {type(it)}, expected type int")
    sorted_arr = iterate_alg(oracle_values, n, mode, it)
    
    print(f"Outputting:")
    if rand_ov:
        name = f"The States of the Grover Search with random Oracle Values {oracle_values}, after {int(it)} iterations is: "
    else:
        name = f"The States of the Grover Search with Oracle Values {oracle_values}, after {int(it)} iterations is: "
    print_out_kets = format_ket_notation(sorted_arr, type="topn", num_bits=int(np.ceil(n)), precision = (3 if n < 20 else 6))
    print(f"{name}\n{print_out_kets}")
    print(f"Total Time to run Grover's Algorithm: {Grover_timer.elapsed()[0]:.4f} seconds")
    print("=" * linewid)
    print("\n")
    

def grover_search(oracle_values, **kwargs):
    n_cap = kwargs.get("n_cap", 16)
    n = kwargs.get("n", None)
    iterations = kwargs.get("iterations", None)
    verbose = kwargs.get("verbose", False)
    mode = kwargs.get("mode", "fast")
    n_rand = kwargs.get("n_rand", 12)
    balanced_param = kwargs.get("balanced_param", 100)
    rand_ov=False
    if isinstance(oracle_values, int):
        rand_ov = oracle_values
    if rand_ov:
        oracle_values = []
        for i in range(rand_ov):
            oracle_values.append(randint(0, 2**n_rand - 1))
        rand_ov = True
    run(oracle_values, n, n_cap, iterations, mode, verbose, rand_ov, balanced_param)