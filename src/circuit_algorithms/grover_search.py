from ..circuit.classes.lightweight_circuit import circuit_lw





def grover_search(oracle_values, **kwargs):
    n_cap = kwargs.get("n_cap", 12)
    n = kwargs.get("n", None)
    iterations = kwargs.get("iterations", None)
    verbose = kwargs.get("verbose", False)





class Grover(StrMixin):                                               #this is the Grover algorithms own class
    """The class to run and analyse Grovers algorithm"""
    array_name = "kets"
    def __init__(self, *args, **kwargs):
        self.fast = kwargs.get("fast", True)
        self.n_cap: int = int(kwargs.get("n_cap",16 if self.fast else 12))         
        self.n = kwargs.get("n", None)
        self.it = kwargs.get("iterations", None)         
        self.oracle_values = []                                             
        self.rand_ov = 0                            
        self.results = kwargs.get("results", [])                            
        self.iter_calc = kwargs.get("iter_calc", None)
        self.balanced_param = kwargs.get("balanced_param", 100)
        for arg in args:
            if isinstance(arg, list): 
                self.oracle_values.extend(arg)    #chooses between using the given oracle values or finding random ones
            elif isinstance(arg, int):
                self.rand_ov = arg
        
    def phase_oracle(self, qub: Qubit, oracle_values: list) -> Qubit:          #the Grover phase oracle
        """Computes the phase flip produced from the oracle values"""
        qub.vector[oracle_values] *= -1                     #flips the values in the index of the oracle values
        return qub
    
    def optimal_iterations(self, n: int) -> tuple[float, int]:
        """Calculates the best number of iterations for a given number of qubits"""
        search_space: int = 2**n
        op_iter: float = (np.pi/4)*np.sqrt((search_space)/len(self.oracle_values)) -1/2        #the standard equation to find the optimal iterations
        return op_iter, search_space
    
    def init_states(self) -> tuple[Qubit, Gate]:
        """Creates the Quantum state and the Hadamard gates needed for the algorithm"""
        timer = Timer()
        qub = Qubit.q0(n=self.n) #initialises the Qubit state |0*n>
        print_array(f"Initialising state {qub.name}")
        if self.fast:               #this is the faster variant of Grovers utilising the FWHT
            print_array(f"Using FWHT to compute {self.n} x {self.n} Hadamard gate application")
            had = FWHT()                #"creates the had for the FWHT, even thought the FWHT isnt a physical gate"
            return qub, had
        else:
            n_had = Hadamard                   
            print_array(f"Initialising {self.n} x {self.n} Hadamard")
            for i in range(self.n-1):    #creates the qubit and also the tensored hadamard for the given qubit size
                n_had **= Hadamard              #if not fast then tensors up a hadamard to the correct size
                print(f"\r{i+2} x {i+2} Hadamard created", end="")    #allows to clear line without writing a custom print function in print_array
            print(f"\r",end="")
            print_array(f"\rHadamard and Quantum State created, time to create was: {timer.elapsed()[0]:.4f}")
            return qub,n_had

    def iterate_alg(self) -> Qubit:
        """The main core algorithm of the program where the phase gates and Hadamard gates are applied"""
        it = 0
        timer = Timer()
        print_array(f"Running FWHT algorithm:") if self.fast else print_array(f"Running algorithm:")
        qub, had = self.init_states()   #this is where the prveious function is called and the states and hadamard are produced
        while it < int(self.it):   #this is where the bulk of the computation actually occurs and is where the algorithm is actually applied
            print(f"\rIteration {it + 1}:                                                                  ", end="")
            if it != 0:
                qub: Qubit = final_state
            print(f"\rIteration {it + 1}: Applying first Hadamard                                          ", end="")
            hadamard_qubit = had * qub       #applies a hadamard to every qubit                           STEP 1
            print(f"\rIteration {it + 1}: Applying phase oracle                                            ", end="")
            oracle_qubit = self.phase_oracle(hadamard_qubit, self.oracle_values)              #STEP 2   phase flips the given oracle values
            print(f"\rIteration {it + 1}: Applying second Hadamard                                         ", end="")
            intermidary_qubit = had * oracle_qubit                                            #STEP 3 Applies the hadamard again
            print(f"\rIteration {it + 1}: Flipping the Qubits phase except first Qubit                     ", end="")
            intermidary_qubit.vector *= -1           #inverts all of the phases of the qubit values             STEP 4a
            intermidary_qubit.vector[0] *= -1              #inverts back the first qubits phase                 STEP 4b
            print(f"\rIteration {it + 1}: Applying third and final Hadamard                                ", end="")
            final_state = had * intermidary_qubit        #applies yet another hadamard gate to the qubits    STEP 5
            it += 1                   #adds to the iteration counter
            print(f"\r                                                                                     Time elapsed:{timer.elapsed()[0]:.4f} secs", end="")
        print(f"\r",end="")
        print_array(f"\rFinal state calculated. Time to iterate algorithm: {timer.elapsed()[1]:.4f} secs                                                                        ")
        return final_state

    def compute_n(self) -> int:
        """Computes the optimal number of qubits and thus search space by finding a good iteration value based on the specific algorithm ran within"""
        if isinstance(self.n_cap, int):                #check on the input type of self.ncap
            print_array(f"Using up to {self.n_cap} Qubits to run the search")
            max_oracle = max(self.oracle_values)             #finds the largest one to find the min qubits
            n_qubit_min = 1
            while max_oracle > 2**n_qubit_min:             #when picking the qubits, we need enough to allow the search space to be bigger than all the oracle values
                n_qubit_min += 1
            if n_qubit_min > self.n_cap:
                raise QC_error(f"The search space needed for this search is larger than the qubit limit {self.n_cap}.")
            if self.it == None:          #if no given iteration then will find it
                print_array(f"No iteration value given, so will now calculate the optimal iterations")
                n_qubit_range = np.arange(n_qubit_min, self.n_cap + 1, dtype=int)
                if self.iter_calc == None or self.iter_calc == "round":          #this is the default type
                    int_val = 0
                    print_array(f"Now computing n for the optimal iteration closest to a whole number")
                    for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                        op_iter = self.optimal_iterations(i)[0]
                        if op_iter >= 1:
                            int_dist: float = op_iter - np.floor(op_iter)  #finds the float value
                            int_dist: float = abs(int_dist-0.5)             #shifts them down so its the distance from any integer
                            print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                            if int_dist > int_val:            #iterates through to find the smallest distance from an integer
                                self.n: int = i
                                int_val: float = int_dist   #is the standard protocol of rounding to the nearest whole number
                    return self.n
                elif self.iter_calc == "floor":              #is the next protocol that floors every value to the nearest whole number and runs with that
                    int_val = 1
                    print_array(f"Now computing n for the optimal iteration closest to the number below it")
                    for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                        op_iter = self.optimal_iterations(i)[0]                 #computes the optimal amount
                        if op_iter >= 1:
                            int_dist: float = op_iter - np.floor(op_iter)  #finds the float value
                            print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                            if int_dist < int_val:            #iterates through to find the smallest distance from an integer
                                self.n: int = i
                                int_val: float = int_dist
                    return self.n
                elif self.iter_calc == "balanced":        #allows for a balanced approach, this is where if the gap "up" to the next iteration is small enough
                    if isinstance(self.balanced_param, int):            #then is will take that value over going down
                        int_val_floor = 1
                        int_val_round = 0
                        print_array(f"Now computing n for the optimal iteration using a balanced algorithm")
                        for i in n_qubit_range:   #goes through the range of possible qubit values from the smallest possible for the given oracle values up to the cap
                            op_iter = self.optimal_iterations(i)[0]
                            if op_iter >= 1:
                                int_dist_floor: float = op_iter - np.floor(op_iter)  #finds the float value
                                int_dist_round: float = 2*abs(int_dist_floor-0.5) 
                                print_array(f"Optimal iterations for {i} Qubits is: {op_iter:.3f}")
                                if int_dist_floor < int_val_floor:            #iterates through to find the smallest distance from an integer
                                    n_floor: int = i
                                    int_val_floor: float = int_dist_floor
                                if int_dist_round > int_val_round:            #iterates through to find the smallest distance from an integer
                                    n_round: int = i
                                    int_val_round: float = int_dist_round
                        if (1-int_val_round) < int_val_floor / self.balanced_param:      #this is where it calcs which one to use
                            self.n = n_round
                            self.iter_calc = "round"
                            print_array(f"The optimal iteration is computed through rounding")
                        else:
                            self.n = n_floor
                            self.iter_calc = "floor"
                            print_array(f"The optimal iteration is computed by flooring")
                        return self.n
                    raise TypeError(f"balanced_param cannot be of type {type(self.balanced_param)}, expected str")
                raise TypeError(f"iter_calc cannot be of type {type(self.iter_calc)}, expected str")
            else:
                self.n = n_qubit_min
                print_array(f"Running the given {self.it} iterations with the minimum number of qubits {self.n}")
                return self.n
        raise QC_error(f"The qubit limit cannot be of {type(self.n_cap)}, expected type int")
    
    def run(self) -> "Grover":     #Grovers algorithm, can input the number of qubits and also a custom amount of iterations
        """This is the function to initiate the search and compiles the other functions together and prints the values out"""
        Grover_timer = Timer()
        if self.rand_ov:
            console.rule(f"Grovers search with random oracle values", style="grover_header")
            self.oracle_values = np.ones(self.rand_ov)    #basically if oraclae values are not given, it will find n random values and use those instead
        else:
            console.rule(f"Grovers search with oracle values: {self.oracle_values}", style="grover_header")
        if self.n == None:               #if the number of qubits required is not given then run:
            self.n = self.compute_n()
            search_space: int = 2**self.n       #computes the final search space for the chosen n
            print_array(f"Using {self.n} Qubits with a search space of {search_space} to get the best accuracy")
        elif isinstance(self.n, int):
            search_space: int = 2**self.n       #computes the search space for the n provided
            print_array(f"Using {self.n} Qubits with a search space of {search_space}")
        else:
            raise QC_error(f"self.n is of the wrong type {type(self.n)}, expected type int")

        if self.rand_ov:
            self.oracle_values = []
            for i in range(self.rand_ov):
                self.oracle_values.append(randint(0, 2**self.n - 1))
            self.rand_ov = self.oracle_values   #computes the random oracle values and replaces oracle values with them
        op_iter = self.optimal_iterations(self.n)[0]
        if self.it == None:     #now picks an iteration value
            if self.iter_calc == "round" or self.iter_calc == None:
                self.it = round(op_iter)
            elif self.iter_calc == "floor":
                self.it = int(np.floor(op_iter))      #picks which algorithm to apply
            elif self.iter_calc == "balanced":
                self.it = int(np.floor(op_iter))
            else:
                raise QC_error(f"Invalid keyword argument {self.iter_calc} of {type(self.iter_calc)}")
            
            if self.it < 1:    #obviously we cant have no iterations so it atleast does 1 iteration
                self.it = 1.0       #more of a failsafe, will almost certainly be wrong as the percentage diff from the it value to 1 will be large?
                print_array(f"Computing 1 iteration, most likely will not be very accurate")
            else:
                print_array(f"Optimal number of iterations are: {op_iter:.3f}")
                print_array(f"Will run {self.it} iterations")
        elif isinstance(self.it, int):
            print_array(f"Number of iterations to perform are: {self.it}")
        else:
            raise QC_error(f"Iterations cannot be of {type(self.it)}, expected type int")
        final_state = self.iterate_alg()
        print_array(f"Computing Probability Distribution of States")
        final_state = Measure(state=final_state, fast=True)
        print_array(f"Finding the probabilities for the top n Probabilities (n is the number of oracle values)")
        sorted_arr = top_probs(final_state.list_probs(), n=len(self.oracle_values))         #finds the n top probabilities
        print_array(f"Outputing:")
        output = Grover(final_state.name, n=self.n, results=sorted_arr)         #creates a Grover instance
        if self.rand_ov:
            output.name = f"The States of the Grover Search with random Oracle Values {self.oracle_values}, after {int(self.it)} iterations is: "
        else:
            output.name = f"The States of the Grover Search with Oracle Values {self.oracle_values}, after {int(self.it)} iterations is: "
        print_array(output)                #prints that Grover instance
        console.rule(f"Total Time to run Grover's Algorithm: {Grover_timer.elapsed()[0]:.4f} seconds", style="grover_header")
        print()
        return output              #returns the value