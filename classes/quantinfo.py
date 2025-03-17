from numpy import np

class QuantInfo:
     
     @staticmethod
     def fidelity(self, rho_1: np.ndarray=None, rho_2: np.ndarray=None) -> float:
        """Computes the fidelity between two states
        Args:
            self: The Density instance
            rho_1: Defaults to self.rho_a, but can take any given rho
            rho_2: Defaults to self.rho_b, but can take any given rho
        Returns:
            self.fidelity_ab: if self.rho_a and self.rho_b are the default
            fidelity: if using custom rho_1 and rho_2"""
        if self.rho_a is not None and rho_1 is None:
            rho_1 = self.rho_a
        if self.rho_b is not None and rho_2 is None:
            rho_2 = self.rho_b
            if isinstance(rho_1, np.ndarray) and isinstance(rho_2, np.ndarray):
                rho_1 = reshape_matrix(rho_1)
                sqrt_rho1: np.ndarray = sqrtm(rho_1)           #has to be 2D to use this function
                flat_sqrt_rho1 = flatten_matrix(sqrt_rho1)     #flattens back down for matrix multiplication
                product =  flat_sqrt_rho1 * rho_2 * flat_sqrt_rho1
                sqrt_product = sqrtm(reshape_matrix(product))             #reshapse and sqrts in the same line
                flat_sqrt_product = flatten_matrix(sqrt_product)           #flattens to take trace of, could be more efficient to use np.trace but i like avoiding external functions where possible
                mat_trace = trace(flat_sqrt_product)
                mat_trace_conj = np.conj(mat_trace)
                fidelity = (mat_trace*mat_trace_conj).real
                if rho_1 is None and rho_2 is None:
                    self.fidelity_ab = fidelity
                    return self.fidelity_ab
                return fidelity
            
    @staticmethod
    def trace_distance(self, rho_1: np.ndarray=None, rho_2: np.ndarray=None) -> float:
        """Computes the trace distance of two states
        Args:
            self: The Density instance
            rho_1: Defaults to self.rho_a, but can take any given rho
            rho_2: Defaults to self.rho_b, but can take any given rho
        Returns:
            self.trace_dist: if self.rho_a and self.rho_b are the default
            trace_dist: if using custom rho_1 and rho_2"""
        if self.rho_a is not None and rho_1 is None:
            rho_1 = self.rho_a
        if self.rho_b is not None and rho_2 is None:
            rho_2 = self.rho_b
        diff_mat = rho_1 - rho_2
        rho_dim = int(np.sqrt(len(rho_1)))
        dim_range = np.arange(rho_dim)
        trace_dist = np.sum(0.5 * np.abs(diff_mat[dim_range + rho_dim * dim_range]))
        if rho_1 is None and rho_2 is None:
            self.trace_dist = trace_dist
            return self.trace_dist
        return trace_dist
        
    @staticmethod    
    def vn_entropy(self, rho: np.ndarray=None) -> float:
        """Computes the Von Neumann entropy of a state
        Args:
            self: The Density instance
            rho: Defaults to self.rho if no rho given
                 Can compute vne for any given rho
        Returns:
            float: The Von Neumann entropy
        """
        if self.rho is not None and rho is None:
            rho = self.rho
        if isinstance(rho, np.ndarray):
            reshaped_rho = reshape_matrix(rho)
            eigenvalues, eigenvectors = np.linalg.eig(reshaped_rho)
            entropy = 0
            for ev in eigenvalues:
                if ev > 0:    #prevents ev=0 which would create an infinity from the log2
                    entropy -= ev * np.log2(ev)
            if entropy < 1e-10:         #rounds the value if very very small
                entropy = 0.0
            return entropy
        raise DensityError(f"No rho matrix provided")


    @staticmethod        
    def shannon_entropy(self, state: Qubit=None) -> float:
        """computes the shannon entropy of a mixed state
        Args:
            self: The density instance
            state: Defaults to self.state if no state given
                   Can compute se for any given state
        Returns:
                float: The Shannon entropy"""
        if self.state is not None and state is None:
            if isinstance(self.state, Qubit):
                state = self.state
            else:
                raise DensityError(f"self.state is of type {type(self.state)}, expected Qubit class")
        if isinstance(state, Qubit) and state.state_type == "mixed":
            entropy = 0
            for weights in state.weights:
                if weights > 0:    #again to stop infinities
                    entropy -= weights * np.log2(weights)
            if entropy < 1e-10:
                entropy = 0.0
            return entropy
        raise DensityError(f"No mixed Quantum state of type Qubit provided")


    @staticmethod        
    def quantum_conditional_entropy(self, rho_a: np.ndarray=None, rho_b: np.ndarray=None) -> float:    #rho is the one that goes first in S(A|B)
        """Computes the quantum conditional entropy of two states
        Args:
            self: The density instance
            rho_a: Defaults to self.rho_a, but can take any given rho
            rho_b: Defaults to self.rho_b, but can take any given rho
        Returns:
            self.cond_ent: if self.rho_a and self.rho_b are the default
            cond_ent: if using custom rho_a and rho_b"""
        if isinstance(self.rho_a, np.ndarray) and isinstance(self.rho_b, np.ndarray):
            if rho_a is None:
                rho_1 = self.rho_a
            if rho_b is None:
                rho_2 = self.rho_b
        else:
            raise DensityError(f"Incorrect type {type(self.rho_a)} and type {type(self.rho_b)}, expeted both numpy arrays")
        cond_ent = self.vn_entropy(rho_1) - self.vn_entropy(rho_2)
        if rho_a is None and rho_b is None:
            self.cond_ent = cond_ent
            return self.cond_ent
        return cond_ent


    @staticmethod            
    def quantum_mutual_info(self) -> float:                   #S(A:B)
        """Computes the quantum mutual information of a system
        Args:
            self: The density instance
                  Takes the internal self.rho, self.rho_a and self.rho_b as inputs
                  self.rho is the overall system while rho_a and rho_b are the traced out components of the system
        Returns:
            float: returns the quantum mutual information of the three rho matrices"""
        if all(isinstance(i, np.ndarray) for i in (self.rho, self.rho_a, self.rho_b)):   #compact way to check all three variables
            mut_info = self.vn_entropy(self.rho_a) + self.vn_entropy(self.rho_b) - self.vn_entropy(self.rho)
            return mut_info
        raise DensityError(f"You need to provide rho a, rho b and rho for this computation to work")


    @staticmethod    
    def quantum_relative_entropy(self, rho_a:np.ndarray=None, rho_b:np.ndarray=None) -> float:   #rho is again the first value in S(A||B)
        """Computes the quantum relative entropy of two Quantum states
        Args:
            self: The density instance
            rho_a: Defaults to self.rho_a, but can take any given rho
            rho_b: Defaults to self.rho_b, but can take any given rho
        Returns:
            float: returns the quantum relative entropy"""
        if isinstance(self.rho_a, np.ndarray) and isinstance(self.rho_b, np.ndarray):
            if rho_a is None:
                rho_a = self.rho_a
            if rho_b is None:
                rho_b = self.rho_b
        if isinstance(rho_a, np.ndarray) and isinstance(rho_b, np.ndarray):
            rho_1 = np.zeros(len(rho_a),dtype=np.complex128)
            rho_2 = np.zeros(len(rho_b),dtype=np.complex128)
            for i, val in enumerate(rho_a):
                rho_1[i] = val
                rho_2[i] += 1e-10 if val == 0 else 0          #regularises it for when the log is taken
            for i, val in enumerate(rho_b):
                rho_1[i] = val
                rho_2[i] += 1e-10 if val == 0 else 0
            quant_rel_ent = trace(rho_1*(flatten_matrix(logm(reshape_matrix(rho_1)) - logm(reshape_matrix(rho_2)))))
            return quant_rel_ent
        raise DensityError(f"Incorrect type {type(self.rho_a)} and type {type(self.rho_b)}, expected both numpy arrays")

    