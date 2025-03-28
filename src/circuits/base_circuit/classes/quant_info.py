import numpy as np
from ...circuit_utilities.circuit_errors import QuantInfoError
from .qubit import Qubit
from .gate import X_Gate, Y_Gate, Z_Gate
from scipy.linalg import sqrtm, logm
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from ...circuit_config import *

__all__ = ["QuantInfo"]


class QuantInfo:


    @classmethod
    def __dir__(cls):
        methods = ["qubit_info", "state_info", "two_state_info", "purity", "linear_entropy", "vn_entropy", "shannon_entropy",
           "quantum_discord", "fidelity", "trace_distance", "trace_distance_bound", "quantum_conditional_entropy",
           "quantum_mutual_information", "quantum_relative_entropy", "bloch_vector", "bloch_plotter"]
        return methods

    @staticmethod
    def qubit_info(qub: Qubit, title=True) -> str:
        print("-" * linewid)
        print(f"Qubit {qub.name} OVERVIEW") if qub.name else print(f"Qubit Overview")
        print(f"Purity of qubit: {QuantInfo.purity(qub):.{p_prec}f}")
        print(f"Linear Entropy of qubit: {QuantInfo.linear_entropy(qub):.{p_prec}f}")
        print(f"Von Neumann entropy of qubit: {QuantInfo.vn_entropy(qub):.{p_prec}f}")
        print(f"Shannon Entropy of qubit: {QuantInfo.shannon_entropy(qub):.{p_prec}f}")
        if qub.dim < 5:
            print(f"Rho matrix of qubit:\n{qub.rho}")
        print("-" * linewid)
        print("\n")

    @staticmethod
    def state_info(state: Qubit, title=True) -> str:
        if title:
            print("-" * linewid)
            print(f"QUANTUM INFORMATION OVERVIEW")
        print(f"Purity of state: {QuantInfo.purity(state):.{p_prec}f}")
        print(f"Linear Entropy of state: {QuantInfo.linear_entropy(state):.{p_prec}f}")
        print(f"Von Neumann Entropy of the whole state: {QuantInfo.vn_entropy(state):.{p_prec}f}")
        print(f"Shannon Entropy of the whole state: {QuantInfo.shannon_entropy(state):.{p_prec}f}")
        print(f"Information on the states individual qubits:")
        print("-" * int(linewid/2))
        for i in range(state.n):
            print(f"Qubit {i}:")
            print("-" * int(linewid/4))
            print(f"Purity of qubit {i}: {QuantInfo.purity(state[i]):.{p_prec}f}")
            print(f"Linear Entropy of qubit {i}: {QuantInfo.linear_entropy(state[i]):.{p_prec}f}")
            print(f"Von Neumann entropy of qubit {i}: {QuantInfo.vn_entropy(state[i]):.{p_prec}f}")
            print(f"Shannon Entropy of qubit: {QuantInfo.shannon_entropy(state[i]):.{p_prec}f}")
        if title:
            print("-" * linewid)

    @staticmethod
    def two_state_info(state_1: Qubit, state_2: Qubit) -> str:
        print("-" * linewid)
        print(f"QUANTUM INFORMATION OF TWO STATES OVERVIEW")
        print(f"\nIndividual Info for {state_1.name}:") if state_1.name else print(f"\nIndividual Info for State 1:")
        print("-" * int(linewid/2))
        QuantInfo.state_info(state_1, title=False)
        print(f"\nIndividual Info for {state_2.name}:") if state_2.name else print(f"\nIndividual Info for State 2:")
        print("-" * int(linewid/2))
        QuantInfo.state_info(state_2, title=False)
        if state_1.name and state_2.name:
            print(f"\nInfo for the states {state_1.name} and {state_2.name} together:")
        else:
            print(f"\nInfo for State 1 and State 2 together:")
        print("-" * int(linewid/2))
        print(f"Fidelity: {QuantInfo.fidelity(state_1, state_2):.{p_prec}f}")
        print(f"Trace Distance: {QuantInfo.trace_distance(state_1, state_2):.{p_prec}f}")
        lower, upper = QuantInfo.trace_distance_bound(state_1, state_2)
        print(f"Trace Distance Bound: {lower:.{p_prec}f} =< Trace Distance =< {upper:.{p_prec}f}")
        print(f"Quantum Conditional Entropy: {QuantInfo.quantum_conditional_entropy(state_1, state_2):.{p_prec}f}")
        print(f"Quantum Mutual Information: {QuantInfo.quantum_mutual_info(state_1, state_2):.{p_prec}f}")
        print(f"Quantum Relative Entropy: {QuantInfo.quantum_relative_entropy(state_1, state_2):.{p_prec}f}")
        print(f"Quantum Discord: {QuantInfo.quantum_discord(state_1, state_2):.{p_prec}f}")
        print("-" * linewid)


    @staticmethod
    def purity(state: Qubit) -> float:      #a measure of mixedness
        return np.einsum('ij,ji', state.rho, state.rho).real     #dot product then trace over
    
    @staticmethod                           #an approximation of von neumann
    def linear_entropy(state: Qubit) -> float:
        return 1 - QuantInfo.purity(state).real
    
    @staticmethod
    def vn_entropy(state: Qubit) -> float:
        if isinstance(state, Qubit):
            N = state.rho.shape[0]
            k = max(1, N - 2)
            eigenvalues, eigenvectors =  eigsh(state.rho, k=k, which="LM") if sparse.issparse(state.rho) else np.linalg.eig(state.rho)
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
        if isinstance(state, Qubit):
            diag_probs = np.diag(state.rho)
            entropy = -np.sum(diag_probs[diag_probs > 0] * np.log2(diag_probs[diag_probs > 0]))
            if entropy < 1e-10:         #rounds the value if very very small
                entropy = 0.0
            return entropy
        raise QuantInfoError(f"No mixed Quantum state of type Qubit provided")

    @staticmethod
    def quantum_discord(state_1: Qubit, state_2: Qubit) -> float:     #measures non classical correlation
        return QuantInfo.quantum_mutual_info(state_1, state_2) - QuantInfo.quantum_conditional_entropy(state_1, state_2)

    @staticmethod
    def fidelity(state_1: Qubit, state_2: Qubit) -> float:
        if state_1 is None or state_2 is None:
            raise QuantInfoError(f"Must provide two states of type Qubit")
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            rho_1 = state_1.rho
            rho_2 = state_2.rho
            product =  sqrtm(rho_1) @ rho_2 @ sqrtm(rho_1)
            sqrt_product = sqrtm(product)
            mat_trace = np.trace(sqrt_product)
            fidelity = (mat_trace*np.conj(mat_trace)).real
            return fidelity
        raise QuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")
    
    @staticmethod
    def trace_distance(state_1: Qubit, state_2: Qubit) -> float:
        diff_state = state_1 - state_2
        eigenvalues = np.linalg.eigvals(diff_state.rho)
        abs_eigenvalues = np.abs(eigenvalues)
        trace_dist = 0.5 * np.sum(abs_eigenvalues)
        return trace_dist
    
    @staticmethod
    def trace_distance_bound(state_1:Qubit, state_2: Qubit) -> tuple[float, float]:
        fidelity = QuantInfo.fidelity(state_1, state_2)
        if np.isclose(fidelity, 1.0, atol=1e-4):
            lower = 0.0
            upper = 0.0
        elif np.isclose(fidelity, 0.0, atol=1e-4):
            lower = 1.0
            upper = 1.0
        else:
            lower = 1 - np.sqrt(fidelity)
            upper = np.sqrt(1 - fidelity)
        return lower, upper
    
    @staticmethod
    def quantum_conditional_entropy(state_1: Qubit, state_2: Qubit) -> float:    #S(A|B)
        cond_ent = QuantInfo.vn_entropy(state_1) - QuantInfo.vn_entropy(state_2)
        return cond_ent
    
    @staticmethod
    def quantum_mutual_info(state_1, state_2) -> float:                               #S(A:B)
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            mut_info = QuantInfo.vn_entropy(state_1) + QuantInfo.vn_entropy(state_2) - QuantInfo.vn_entropy(state_1 % state_2)
            return mut_info
        raise QuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")

    @staticmethod
    def quantum_relative_entropy(state_1: Qubit, state_2: Qubit) -> float:   #rho is again the first value in S(A||B)
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            rho_1 = state_1.rho
            rho_2 = state_2.rho
            eigenvalues_1, eigenvectors_1 = np.linalg.eig(rho_1)
            eigenvalues_2, eigenvectors_2 = np.linalg.eig(rho_2)
            threshold = 1e-10
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
    
    @staticmethod
    def bloch_vector(qubit: Qubit) -> np.ndarray:
        if not isinstance(qubit, Qubit):
            raise QuantInfoError(f"qubit cannot be of type {type(qubit)}, expected type Qubit")
        if qubit.n == 1:
            x = np.real(np.trace((qubit | X_Gate).rho))
            y = np.real(np.trace((qubit | Y_Gate).rho))
            z = np.real(np.trace((qubit | Z_Gate).rho))
            return np.array([x, y, z])
        raise QuantInfoError(f"Can only visualise single qubit states in a Bloch Sphere repr")

    @staticmethod
    def bloch_plotter(qubit: Qubit) -> None:                   #only for a single qubit
        """A bloch plotter that can plot a single Qubit on the bloch sphere with Matplotlib"""
        plot_counter = 0
        x, y, z = QuantInfo.bloch_vector(qubit)
        ax = plt.axes(projection="3d")
        ax.quiver(0,0,0,x,y,z)
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x_sp = np.cos(u)*np.sin(v)
        y_sp = np.sin(u)*np.sin(v)
        z_sp = np.cos(v)
        ax.set_xlabel("X_Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Bloch Sphere")
        ax.plot_surface(x_sp, y_sp, z_sp, color="g", alpha=0.3)
        ax.axes.grid(axis="x")
        ax.text(0,0,1,"|0>")
        ax.text(0,0,-1,"|1>")
        ax.text(1,0,0,"|+>")
        ax.text(-1,0,0,"|->")
        ax.text(0,1,0,"|i>")
        ax.text(0,-1,0,"|-i>")
        ax.plot([-1,1],[0,0],color="black")
        ax.plot([0,0],[-1,1],color="black")
        ax.plot([0,0],[-1,1],zdir="y",color="black")
        plot_counter += 1