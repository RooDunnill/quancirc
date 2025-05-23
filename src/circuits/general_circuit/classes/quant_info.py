import numpy as np
import logging
from ...circuit_config import *
from ..utilities.circuit_errors import QuantInfoError
from .qubit import Qubit
from .gate import X_Gate, Y_Gate, Z_Gate
from scipy.linalg import sqrtm, logm
from scipy import sparse
from scipy.sparse.linalg import eigsh, eigs
import matplotlib.pyplot as plt
from ...circuit_utilities.sparse_funcs import *

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
        logging.info("-" * linewid)
        logging.info(f"Qubit {qub.id} OVERVIEW")
        logging.info(f"Purity of qubit: {QuantInfo.purity(qub):.{p_prec}f}")
        logging.info(f"Linear Entropy of qubit: {QuantInfo.linear_entropy(qub):.{p_prec}f}")
        logging.info(f"Von Neumann entropy of qubit: {QuantInfo.vn_entropy(qub):.{p_prec}f}")
        logging.info(f"Shannon Entropy of qubit: {QuantInfo.shannon_entropy(qub):.{p_prec}f}")
        if qub.dim < 5:
            logging.info(f"Rho matrix of qubit:\n{qub.rho}")
        logging.info("-" * linewid)
        logging.info("\n")

    @staticmethod
    def state_info(state: Qubit, title=True) -> str:
        if title:
            logging.info("-" * linewid)
            logging.info(f"QUANTUM INFORMATION OVERVIEW")
        logging.info(f"Purity of state: {QuantInfo.purity(state):.{p_prec}f}")
        logging.info(f"Linear Entropy of state: {QuantInfo.linear_entropy(state):.{p_prec}f}")
        logging.info(f"Von Neumann Entropy of the whole state: {QuantInfo.vn_entropy(state):.{p_prec}f}")
        logging.info(f"Shannon Entropy of the whole state: {QuantInfo.shannon_entropy(state):.{p_prec}f}")
        logging.info(f"Information on the states individual qubits:")
        logging.info("-" * int(linewid/2))
        for i in range(state.n):
            logging.info(f"Qubit {i}:")
            logging.info("-" * int(linewid/4))
            logging.info(f"Purity of qubit {i}: {QuantInfo.purity(state[i]):.{p_prec}f}")
            logging.info(f"Linear Entropy of qubit {i}: {QuantInfo.linear_entropy(state[i]):.{p_prec}f}")
            logging.info(f"Von Neumann entropy of qubit {i}: {QuantInfo.vn_entropy(state[i]):.{p_prec}f}")
            logging.info(f"Shannon Entropy of qubit: {QuantInfo.shannon_entropy(state[i]):.{p_prec}f}")
        if title:
            logging.info("-" * linewid)

    @staticmethod
    def two_state_info(state_1: Qubit, state_2: Qubit) -> str:
        logging.info("-" * linewid)
        logging.info(f"QUANTUM INFORMATION OF TWO STATES OVERVIEW")
        logging.info(f"\nIndividual Info for {state_1.id}:")
        logging.info("-" * int(linewid/2))
        QuantInfo.state_info(state_1, title=False)
        logging.info(f"\nIndividual Info for {state_2.id}:")
        logging.info("-" * int(linewid/2))
        QuantInfo.state_info(state_2, title=False)
        logging.info(f"\nInfo for the states {state_1.id} and {state_2.id} together:")
        logging.info("-" * int(linewid/2))
        logging.info(f"Fidelity: {QuantInfo.fidelity(state_1, state_2):.{p_prec}f}")
        logging.info(f"Trace Distance: {QuantInfo.trace_distance(state_1, state_2):.{p_prec}f}")
        lower, upper = QuantInfo.trace_distance_bound(state_1, state_2)
        logging.info(f"Trace Distance Bound: {lower:.{p_prec}f} =< Trace Distance =< {upper:.{p_prec}f}")
        logging.info(f"Quantum Conditional Entropy: {QuantInfo.quantum_conditional_entropy(state_1, state_2):.{p_prec}f}")
        logging.info(f"Quantum Mutual Information: {QuantInfo.quantum_mutual_info(state_1, state_2):.{p_prec}f}")
        logging.info(f"Quantum Relative Entropy: {QuantInfo.quantum_relative_entropy(state_1, state_2):.{p_prec}f}")
        logging.info(f"Quantum Discord: {QuantInfo.quantum_discord(state_1, state_2):.{p_prec}f}")
        logging.info("-" * linewid)


    @staticmethod
    def purity(state: Qubit) -> float:      #a measure of mixedness
        if sparse.issparse(state.rho):
            trace_value = 0
            for i, j in zip(*state.rho.nonzero()):
                trace_value += state.rho[i, j] * state.rho[j, i]
            return trace_value.real
        else:
            return np.trace(state.rho @ state.rho).real
    
    
    @staticmethod                           #an approximation of von neumann
    def linear_entropy(state: Qubit) -> float:
        return 1 - QuantInfo.purity(state).real
    
    @staticmethod
    def vn_entropy(state: Qubit) -> float:
        if isinstance(state, Qubit):
            if state.n > 14:
                raise QuantInfoError(f"Matrix too big")
            k = state.n * 2
            if state.dim < eig_threshold:
                eigenvalues, eigenvectors =  np.linalg.eig(dense_mat(state.rho))
            else:
                eigenvalues, eigenvectors =  eigsh(state.rho, k=k, which="LM", ncv=k+2) if sparse.issparse(state.rho) else np.linalg.eig(state.rho)
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
            if sparse.issparse(state.rho):
                diag_probs = state.rho.diagonal()
            else:
                diag_probs = np.diag(state.rho) 
            entropy = -np.sum(diag_probs[diag_probs > 0] * np.log2(diag_probs[diag_probs > 0]))
            if entropy < 1e-10:         #rounds the value if very very small
                entropy = 0.0
            return entropy
        raise QuantInfoError(f"No mixed Quantum state of type Qubit provided")

    @staticmethod
    def quantum_discord(state_1: Qubit, state_2: Qubit) -> float:     #measures non classical correlation
        if state_1.n + state_2.n > 14:
            raise QuantInfoError(f"Matrix too big")
        return QuantInfo.quantum_mutual_info(state_1, state_2) - QuantInfo.quantum_conditional_entropy(state_1, state_2)

    @staticmethod
    def fidelity(state_1: Qubit, state_2: Qubit) -> float:
        if state_1.rho.shape != state_2.rho.shape:
            raise QuantInfoError(f"The shapes of the matrices must be the same, not {state_1.rho.shape} and {state_2.rho.shape}")
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            rho_1 = dense_mat(state_1.rho)
            rho_2 = dense_mat(state_2.rho)
            product =  sqrtm(rho_1) @ rho_2 @ sqrtm(rho_1)
            sqrt_product = sqrtm(product)
            mat_trace = np.trace(sqrt_product)
            fidelity = (mat_trace*np.conj(mat_trace)).real
            return fidelity
        raise QuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")
    
    @staticmethod
    def trace_distance(state_1: Qubit, state_2: Qubit) -> float:
        if state_1.rho.shape != state_2.rho.shape:
            raise QuantInfoError(f"The shapes of the matrices must be the same, not {state_1.rho.shape} and {state_2.rho.shape}")
        diff_rho = dense_mat(state_1.rho) - dense_mat(state_2.rho)
        eigenvalues = np.linalg.eigvals(diff_rho)
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
            state_1.rho = sparse_mat(state_1.rho)
            state_2.rho = sparse_mat(state_2.rho)
            mut_info = QuantInfo.vn_entropy(state_1) + QuantInfo.vn_entropy(state_2) - QuantInfo.vn_entropy(state_1 % state_2)
            return mut_info
        raise QuantInfoError(f"state_1 and state_2 must both be of type Qubit, not of type {type(state_1)} and {type(state_2)}")

    @staticmethod
    def quantum_relative_entropy(state_1: Qubit, state_2: Qubit) -> float:   #rho is again the first value in S(A||B)
        if isinstance(state_1, Qubit) and isinstance(state_2, Qubit):
            if sparse.issparse(state_1.rho) or sparse.issparse(state_2.rho):
                state_1.rho = sparse_mat(state_1.rho)
                state_2.rho = sparse_mat(state_2.rho)
                if state_1.rho.shape != state_2.rho.shape:
                    raise QuantInfoError(f"The shapes of the matrices must be the same, not {state_1.rho.shape} and {state_2.rho.shape}")
                k = state_1.n * 2
                if state_1.dim < eig_threshold:
                    eigenvalues_1, eigenvectors_1 =  np.linalg.eig(dense_mat(state_1.rho))
                    eigenvalues_2, eigenvectors_2 =  np.linalg.eig(dense_mat(state_2.rho))
                else:
                    eigenvalues_1, eigenvectors_1 =  eigs(state_1.rho, k=k, which="LM", ncv=k+2)
                    eigenvalues_2, eigenvectors_2 =  eigs(state_2.rho, k=k, which="LM", ncv=k+2)

                eigenvalues_1 = np.maximum(eigenvalues_1, 1e-4)
                eigenvalues_2 = np.maximum(eigenvalues_2, 1e-4)
                log_eigenvalues_1 = np.log(eigenvalues_1)
                log_eigenvalues_2 = np.log(eigenvalues_2)
                rho_1_log = eigenvectors_1 @ sparse.diags(log_eigenvalues_1) @ eigenvectors_1.T
                rho_2_log = eigenvectors_2 @ sparse.diags(log_eigenvalues_2) @ eigenvectors_2.T
                rho_1_log = sparse_mat(rho_1_log)
                rho_2_log = sparse_mat(rho_2_log)
                try:
                    quant_rel_ent = (rho_1_log @ state_1.rho - rho_2_log @ state_2.rho).diagonal().sum()  #not correct
                    if quant_rel_ent < 1.e-4:
                        quant_rel_ent = 0.0
                    return quant_rel_ent.real
                except Exception as e:
                    raise QuantInfoError(f"Error in computing relative entropy: {e}")
            else:
                eigenvalues_1, eigenvectors_1 = np.linalg.eig(state_1.rho)
                eigenvalues_2, eigenvectors_2 = np.linalg.eig(state_2.rho)
                eigenvalues_1 = np.maximum(eigenvalues_1, 1e-4)
                eigenvalues_2 = np.maximum(eigenvalues_2, 1e-4)
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