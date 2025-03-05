def list_proj_probs(self, qubit: int = None, povm: list = None) -> np.ndarray:
    if povm is not None:
        # Non-projective measurements (POVMs)
        probs = np.array([np.real(np.trace(E @ self.density.rho)) for E in povm], dtype=np.float64)
        return probs
    
    if qubit is None:
        # Default full projection probabilities
        if self.fast:
            vector = self.state.vector
            return np.real(np.multiply(vector, np.conj(vector)))
        elif self.measurement_state == "all" and isinstance(self.density, Density):
            if self.rho is None:
                self.rho = self.density.rho
            return np.array([self.rho[i + i * self.dim].real for i in range(self.dim)], dtype=np.float64)
        else:
            raise QC_error(qc_dat.error_kwargs)
    else:
        # Single-qubit measurement probabilities (computational basis)
        probs = np.zeros(2)
        for i in range(self.dim):
            binary = bin(i)[2:].zfill(int(np.log2(self.dim)))
            probs[int(binary[-(qubit + 1)])] += self.rho[i + i * self.dim].real if isinstance(self.density, Density) else np.abs(self.state.vector[i]) ** 2
        return probs

def proj_measure_state(self, qubit: int = None, povm: list = None) -> str:
    PD = self.list_proj_probs(qubit, povm)
    measurement = choices(range(len(PD)), weights=PD)[0]
    if povm is not None:
        return f"Measured POVM outcome: {measurement}"
    
    if qubit is None:
        num_bits = int(np.ceil(np.log2(self.dim)))
        result = f"Measured the state: |{bin(measurement)[2:].zfill(num_bits)}>"
        self.state.vector[measurement] = 1.0
        self.state.vector = self.state.vector / np.linalg.norm(self.state.vector)
        return result
    else:
        # Collapse single qubit
        num_bits = int(np.log2(self.dim))
        mask = 1 << qubit
        collapsed = np.zeros_like(self.state.vector, dtype=np.complex128)
        for i in range(self.dim):
            if (i & mask) >> qubit == measurement:
                collapsed[i] = self.state.vector[i]
        self.state.vector = collapsed / np.linalg.norm(collapsed)
        return f"Measured qubit {qubit} in state: |{measurement}>"
