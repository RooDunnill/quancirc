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
        new_rho = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        rho_dim = len(rho)
        rho_n = int(np.log2(rho_dim))
        if trace_out_state_size == rho_n:
            kwargs.update(copy_qubit_attr(self))
            return Qubit(**kwargs)
        traced_out_dim: int = 2**trace_out_state_size
        reduced_dim = int(rho_dim / traced_out_dim)
        new_rho = np.zeros((reduced_dim, reduced_dim),dtype=np.complex128)
        traced_out_dim_range = np.arange(traced_out_dim)
        if trace_out_system == "B":
                for k in range(reduced_dim):
                    for i in range(reduced_dim):           #the shapes of tracing A and B look quite different but follow a diagonalesc pattern
                        new_rho[i, k] = np.sum(rho[traced_out_dim_range+i*traced_out_dim, traced_out_dim_range+k*traced_out_dim])
        elif trace_out_system == "A":
            for k in range(reduced_dim):
                for i in range(reduced_dim):
                    new_rho[i, k] = np.sum(rho[reduced_dim*traced_out_dim_range+i, reduced_dim *traced_out_dim_range+k])
        kwargs = {"rho": new_rho}
        kwargs.update(copy_qubit_attr(self))
        return Qubit(**kwargs)

def isolate_qubit(self, qubit_index: int) -> "Qubit":
        """Used primarily in __getitem__ to return a single Qubit from a multiqubit state, returns a Qubit object"""
        if qubit_index is not None and isinstance(qubit_index, int):
            if qubit_index > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
            if qubit_index == 0:
                isolated_rho = self.partial_trace("B", self.n - 1)
            elif qubit_index == self.n - 1:
                isolated_rho = self.partial_trace("A", self.n - 1)
            elif isinstance(qubit_index, int):
                A_rho = self.partial_trace("B", self.n - qubit_index - 1)
                A_n = int(np.log2(len(A_rho.rho)))
                isolated_rho = self.partial_trace("A", A_n - 1, rho=A_rho.rho)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit_index)}, expected int") 
            return isolated_rho
        
    def decompose_state(self, qubit_index: int) -> tuple["Qubit", "Qubit", "Qubit"]:
        """Used primarily in __setitem__ to 'pull out' the Qubit to be replaced, returns three Qubit objects that can be recombined"""
        if qubit_index is not None:
            if qubit_index > self.n - 1:
                raise QuantumStateError(f"The chosen qubit {qubit_index}, must be no more than the number of qubits in the state: {self.n}")
            if isinstance(qubit_index, int):
                temp_rho = self.partial_trace("B", self.n - qubit_index - 1)
                A_rho = self.partial_trace("B", self.n - qubit_index)
                B_rho = self.partial_trace("A", qubit_index + 1)
                temp_n = int(np.log2(len(temp_rho.rho)))
                isolated_rho = self.partial_trace("A", temp_n - 1, rho=temp_rho.rho)
            else:
                raise QuantumStateError(f"Inputted qubit cannot be of type {type(qubit_index)}, expected int") 
            return A_rho, isolated_rho, B_rho
        
        def partial_trace(self, size_a, size_c, **kwargs):
        rho = kwargs.get("rho", self.rho)
        dim_a = int(2**size_a)
        dim_c = int(2**size_c)
        rho_dim = len(rho)
        dim_b = int(rho_dim/(dim_a*dim_c))
        if size_c == 0:
            rho = np.kron(rho , q0.rho)
            dim_c = 2
        elif size_a == 0:
            rho = np.kron(q0.rho , rho)
            dim_a = 2
        
        rho_reshape = rho.reshape(dim_a, dim_b, dim_c, dim_a, dim_b, dim_c)
        new_rho = np.einsum("abcdef->be", rho_reshape)
            
        print(new_rho)
        print(np.trace(new_rho))
        #new_rho = new_rho / np.trace(new_rho)
        kwargs = {"rho": new_rho}
        kwargs.update(copy_qubit_attr(self))
        return Qubit(**kwargs)