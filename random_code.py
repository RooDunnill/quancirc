def partial_trace(self, trace_out_system: str, trace_out_state_size: int, **kwargs) -> "Qubit":
        """Computes the partial trace of a state, can apply a trace from either 'side' and can trace out an arbitrary amount of qubits
        Args:
            self: The density instance
            **kwargs
            trace_out_system:str : Chooses between A and B which to trace out, defaults to B
            state_size:int : Chooses the number of Qubits in the trace out state, defaults to 1 Qubit
        Returns:self.rho_a if trace_out_system = B
                self.rho_b if trace_out_system = A"""
        rho = kwargs.get("rho", self.rho)
        if not isinstance(rho, (np.ndarray, list)):
            raise QuantumStateError(f"rho cannot be of type {type(rho)}, expected type np.ndarray or type list")
        if trace_out_system not in ["A", "B"]:
            raise QuantumStateError(f"trace_out_system must be either str: 'A' or 'B', cannot be {trace_out_system}")
        if not isinstance(trace_out_state_size, int):
            raise QuantumStateError(f"trace_out_state_size cannot be of type {type(trace_out_state_size)}, expected type int")
        rho_dim = len(rho)
        rho_n = int(np.log2(rho_dim))
        if trace_out_state_size == rho_n:
            return Qubit()
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        new_mat = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        traced_out_dim_range = np.arange(traced_out_dim)
        if isinstance(rho, np.ndarray):
            kwargs = vars(self).copy()
            if trace_out_system == "B":
        # Reshape to group indices for the system to be traced out
                rho_reshaped = rho.reshape(reduced_dim, traced_out_dim, reduced_dim, traced_out_dim)
                new_mat = np.einsum("ikjk->ij", rho_reshaped)  # Sum over the traced-out subsystem
            elif trace_out_system == "A":
                rho_reshaped = rho.reshape(traced_out_dim, reduced_dim, traced_out_dim, reduced_dim)
                new_mat = np.einsum("kilj->ij", rho_reshaped) 
            return Qubit(rho=new_mat)
        
def partial_trace(self, trace_out_system: str, trace_out_state_size: int, **kwargs) -> "Qubit":
        """Computes the partial trace of a state, can apply a trace from either 'side' and can trace out an arbitrary amount of qubits
        Args:
            self: The density instance
            **kwargs
            trace_out_system:str : Chooses between A and B which to trace out, defaults to B
            state_size:int : Chooses the number of Qubits in the trace out state, defaults to 1 Qubit
        Returns:self.rho_a if trace_out_system = B
                self.rho_b if trace_out_system = A"""
        rho = kwargs.get("rho", self.rho)
        if not isinstance(rho, (np.ndarray, list)):
            raise QuantumStateError(f"rho cannot be of type {type(rho)}, expected type np.ndarray or type list")
        if trace_out_system not in ["A", "B"]:
            raise QuantumStateError(f"trace_out_system must be either str: 'A' or 'B', cannot be {trace_out_system}")
        if not isinstance(trace_out_state_size, int):
            raise QuantumStateError(f"trace_out_state_size cannot be of type {type(trace_out_state_size)}, expected type int")
        rho_dim = len(rho)
        rho_n = int(np.log2(rho_dim))
        if trace_out_state_size == rho_n:
            return Qubit()
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        new_mat = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        traced_out_dim_range = np.arange(traced_out_dim)
        if isinstance(rho, np.ndarray):
            kwargs = vars(self).copy()
            if trace_out_system == "B":
                for k in range(reduced_dim):
                    for i in range(reduced_dim):           #the shapes of tracing A and B look quite different but follow a diagonalesc pattern
                        new_mat[i, k] = np.sum(rho[traced_out_dim_range+i*traced_out_dim, traced_out_dim_range+k*traced_out_dim])
                return Qubit(rho=new_mat)
            elif trace_out_system == "A":
                for k in range(reduced_dim):
                    for i in range(reduced_dim):
                        new_mat[i, k] = np.sum(rho[reduced_dim*traced_out_dim_range+i, reduced_dim *traced_out_dim_range+k])
                return Qubit(rho=new_mat)
        raise QuantumStateError(f"self.rho cannot be of type {type(self.rho)}, expected type np.ndarray")