import numpy as np
from utilities import QuantInfoError
from utilities.print_settings import p_prec, linewid
from .qubit import Qubit
from scipy.linalg import sqrtm, logm

class QuantInfo:
     
    @staticmethod
    def state_info(state: Qubit, title=True) -> str:
        if title:
            print("-" * linewid)
            print(f"QUANTUM INFORMATION OVERVIEW")
        print(f"Purity of state: {QuantInfo.purity(state):.{p_prec}f}")
        print(f"Von Neumann Entropy of the whole state: {QuantInfo.vn_entropy(state):.{p_prec}f}")
        for i in range(state.n):
            print(f"Von Neumann entropy of qubit {i}: {QuantInfo.vn_entropy(state[i]):.{p_prec}f}")
        if state.state_type == "mixed":
            print(f"Shannon Entropy: {QuantInfo.shannon_entropy(state):.{p_prec}f}")
        if title:
            print("-" * linewid)

    @staticmethod
    def two_state_info(state_1: Qubit, state_2: Qubit) -> str:
        print("-" * linewid)
        print(f"QUANTUM INFORMATION OF TWO STATES OVERVIEW")
        print(f"\nIndividual Info for {state_1.name}:")
        print("-" * int(linewid/2))
        QuantInfo.state_info(state_1, title=False)
        print(f"\nIndividual Info for {state_2.name}:")
        print("-" * int(linewid/2))
        QuantInfo.state_info(state_2, title=False)
        print(f"\nInfo for the states together:")
        print("-" * int(linewid/2))
        print(f"Fidelity: {QuantInfo.fidelity(state_1, state_2):.{p_prec}f}")
        print(f"Trace Distance: {QuantInfo.trace_distance(state_1, state_2):.{p_prec}f}")
        print(f"Quantum Conditional Entropy: {QuantInfo.quantum_conditional_entropy(state_1, state_2):.{p_prec}f}")
        print(f"Quantum Mutual Information: {QuantInfo.quantum_mutual_info(state_1, state_2):.{p_prec}f}")
        print(f"Quantum Relative Entropy: {QuantInfo.quantum_relative_entropy(state_1, state_2):.{p_prec}f}")
        print("-" * linewid)


    @staticmethod
    def purity(state: Qubit) -> float:
        return np.trace(np.dot(state.rho, state.rho))

    @staticmethod
    def fidelity(state_1: Qubit, state_2: Qubit) -> float:
        if state_1 is None or state_2 is None:
            raise QuantInfoError(f"Must provide two states of type Qubit")
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            rho_1 = state_1.rho
            rho_2 = state_2.rho
            product =  sqrtm(rho_1) * rho_2 * sqrtm(rho_1)
            sqrt_product = sqrtm(product)
            mat_trace = np.trace(sqrt_product)
            fidelity = (mat_trace*np.conj(mat_trace)).real
            return fidelity
        raise QuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")
    
    @staticmethod
    def trace_distance(state_1: Qubit, state_2: Qubit) -> float:
        diff_state = state_1 - state_2
        dim_range = np.arange(state_1.dim)
        trace_dist = np.sum(0.5 * np.abs(diff_state.rho[dim_range, dim_range]))
        return trace_dist
      
    @staticmethod
    def vn_entropy(state: Qubit) -> float:
        if isinstance(state, Qubit):
            rho = state.rho
            eigenvalues, eigenvectors = np.linalg.eig(rho)
            entropy = 0
            for ev in eigenvalues:
                if ev > 0:    #prevents ev=0 which would create an infinity from the log2
                    entropy -= ev * np.log2(ev)
            if entropy < 1e-10:         #rounds the value if very very small
                entropy = 0.0
            return entropy
        raise QuantInfo(f"State cannot be of type {type(state)}, must be of type Qubit")
    
    @staticmethod
    def shannon_entropy(state: Qubit) -> float:
        if isinstance(state, Qubit) and state.state_type == "mixed":
            entropy = 0
            for weights in state.weights:
                if weights > 0:    #again to stop infinities
                    entropy -= weights * np.log2(weights)
            if entropy < 1e-10:
                entropy = 0.0
            return entropy
        raise QuantInfoError(f"No mixed Quantum state of type Qubit provided")
    
    @staticmethod
    def quantum_conditional_entropy(state_1: Qubit, state_2: Qubit) -> float:    #rho is the one that goes first in S(A|B)
        cond_ent = QuantInfo.vn_entropy(state_1) - QuantInfo.vn_entropy(state_2)
        return cond_ent
    
    @staticmethod
    def quantum_mutual_info(state_1, state_2) -> float:                   #S(A:B)
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            mut_info = QuantInfo.vn_entropy(state_1) + QuantInfo.vn_entropy(state_2) - QuantInfo.vn_entropy(state_1 @ state_2)
            return mut_info
        raise QuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")

    @staticmethod
    def quantum_relative_entropy(state_1: Qubit, state_2: Qubit) -> float:   #rho is again the first value in S(A||B)
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            rho_1 = state_1.rho
            rho_2 = state_2.rho
            eigenvalues_1, eigenvectors_1 = np.linalg.eig(rho_1)
            eigenvalues_2, eigenvectors_2 = np.linalg.eig(rho_2)
            threshold = 1e-15
            eigenvalues_1 = np.maximum(eigenvalues_1, threshold)
            eigenvalues_2 = np.maximum(eigenvalues_2, threshold)
            rho_1 = eigenvectors_1 @ np.diag(eigenvalues_1) @ eigenvectors_1.T
            rho_2 = eigenvectors_2 @ np.diag(eigenvalues_2) @ eigenvectors_2.T
            try:
                quant_rel_ent = np.trace(rho_1 @ (logm(rho_1) - logm(rho_2)))
                return quant_rel_ent.real
            except Exception as e:
                raise QuantInfoError(f"Error in computing relative entropy: {e}")
        raise QuantInfoError(f"Incorrect type {type(state_1)} and type {type(state_2)}, expected both Qubit types")
    
    