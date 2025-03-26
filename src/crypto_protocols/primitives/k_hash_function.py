import numpy as np
from time import perf_counter
from random import randint
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from ..crypto_utilities.crypto_errors import PrimitiveError
from .r_key_gen import n_length_int_key

__all__ = ["Hash"]

class Hash:
    plot_count = 0
    p_plot = []
    titles_plot = []
    k_plot = []
    key_plot = []
    plot_all_counts = []
    hash_type = []
    shannon_plot = []
    fig = None
    axs = None
    def __init__(self, **kwargs):
        self.p = kwargs.get("p", 7)
        self.k = int(kwargs.get("k", 6))
        self.key = int(kwargs.get("key", 382956234980))
        self.key_length = int(len(str(self.key)))
        if self.key_length % self.k == 0:
            string = str(self.key)
            interval = int(self.key_length/self.k)
            self.parameters = [int(string[i:i+interval]) for i in range(0, len(string), interval)]
        else:
            raise PrimitiveError(f"The length of the key must be an integer value of the length of k, where k is the k independent value")
        self.name = kwargs.get("name", None)
        self.type = kwargs.get("type", None)
        
    @staticmethod
    def shannon_entropy(bins):
        bins = np.array(bins)
        total = np.sum(bins)
        if total == 0:
            return 0 
        
        probs = bins / total
        
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log2(probs))
        return entropy


    def k_hash(self, vals, text=True):
        self.hash_type = "k_hash"
        start_time = perf_counter()
        def compute_hash(val):
            print(f"\rHashing value: {val} ", end="")
            poly = len(self.parameters)
            hash_value = 0
            poly_change = poly
            for i in self.parameters:
                hash_value += i*(val**poly_change) % self.p
                poly_change -= 1
            hash_value = hash_value % self.p
            return hash_value
        
        if isinstance(vals, int) or isinstance(vals, float):
            hash_value = compute_hash(vals)
            if text:
                print(f"The {self.k} independent hashed value of {vals} is: {hash_value}")
                end_time = perf_counter()
                print(f"Time taken to hash value was: {end_time - start_time:.6f} seconds")
            return hash_value
        
        elif isinstance(vals, np.ndarray) or isinstance(vals, list):
            hash_list = np.array([compute_hash(i) for i in vals])
            if text:
                print(f"The list of {self.k} independent hash inputs and outputs are:\n {np.array(list(zip(vals, hash_list)))}")
                end_time = perf_counter()
                print(f"Time taken to hash list was: {end_time - start_time:.6f} seconds")
            return hash_list
        else:
            raise PrimitiveError(f"The hash input value cannot by of {type(vals)}, it must be a float or an integer.")
        
    def custom_hash(self, vals, text=True):
        self.hash_type = "custom"
        start_time = perf_counter()
        def compute_hash(val):
            print(f"\rHashing value: {val} ", end="")
            poly = len(self.parameters)
            hash_value = 0
            poly_change = poly
            for i in self.parameters:
                hash_value += i*(val**poly_change) % (self.p + i**2 + 1)
                poly_change -= 1
                hash_value = hash_value % self.p
                return hash_value
            return hash_value
        if isinstance(vals, int) or isinstance(vals, float):
            hash_value = compute_hash(vals)
            if text:
                print(f"The {self.k} independent hashed value of {vals} is: {hash_value}")
                end_time = perf_counter()
                print(f"Time taken to hash value was: {end_time - start_time:.6f} seconds")
            return hash_value
        
        elif isinstance(vals, np.ndarray) or isinstance(vals, list):
            hash_list = np.array([compute_hash(i) for i in vals])
            if text:
                print(f"The list of {self.k} independent hash inputs and outputs are:\n {np.array(list(zip(vals, hash_list)))}")
                end_time = perf_counter()
                print(f"Time taken to hash list was: {end_time - start_time:.6f} seconds")
            return hash_list
        else:
            raise PrimitiveError(f"The hash input value cannot by of {type(vals)}, it must be a float or an integer.")
    
    def analyse(self, hashed_vals):
        Hash.plot_count += 1
        cmap = plt.get_cmap("Dark2")
        counts = np.zeros(self.p) 
        for val in hashed_vals:
            counts[val] += 1 
        Hash.plot_all_counts.append(counts)
        Hash.k_plot.append(self.k)
        Hash.p_plot.append(self.p)
        Hash.key_plot.append(self.key)
        Hash.hash_type.append(self.hash_type)
        self.shan_ent = Hash.shannon_entropy(counts)
        Hash.shannon_plot.append(self.shan_ent)
        if Hash.fig is not None:
            plt.close(Hash.fig)
        cols = 3
        rows = (Hash.plot_count + cols - 1) // cols
        if Hash.plot_count == 1:
            Hash.fig, Hash.axs = plt.subplots(1, 1, figsize=(6, 6))
            Hash.axs = [Hash.axs]
        else:
            Hash.fig, Hash.axs = plt.subplots(rows, cols, figsize=(14, 7))
            Hash.axs = Hash.axs.flatten()
            
        for i in range(Hash.plot_count):
            norm = PowerNorm(gamma=50, vmin=0.8, vmax=1)
            entropy_norm = Hash.shannon_plot[i] / np.log2(Hash.p_plot[i])  
            color = cmap(norm(entropy_norm))
            Hash.axs[i].bar(range(Hash.p_plot[i]), Hash.plot_all_counts[i], color=color, edgecolor='black', label=f"Shannon Entropy: {Hash.shannon_plot[i]:.5f}")
            Hash.axs[i].plot([0, 1], [0, 1], label=f"Mod Value: {Hash.p_plot[i]}")
            Hash.axs[i].set_title(self.generate_title(i))
            Hash.axs[i].set_xlabel('Bins')
            Hash.axs[i].legend()
            Hash.axs[i].grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
            if i % cols == 1:
                Hash.axs[0].set_ylabel('Frequency')
        Hash.fig.tight_layout()
        Hash.fig.canvas.draw()
        Hash.fig.suptitle('Hash Function Analysis', fontsize=14, fontweight='bold')
        Hash.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.5, wspace=0.15)
        return counts

    def get_key(self, text=False):
        if text == True:
            print(f"The key for this hash function is: {self.key}")
        return self.key

    def get_k(self, text=False):
        if text == True:
            print(f"The hash function is {self.k} independent")
        return self.k

    def get_p(self, text=False):
        if text == True:
            print(f"The mod value of this function is: {self.p}")
        return self.p

    def generate_title(self, i):
        if Hash.hash_type[i] == "k_hash":
            return f"{i + 1}. {Hash.k_plot[i]}-indep hash values\n with a max SE of {np.log2(self.p):.3f}"
        elif Hash.hash_type[i] == "custom":
            return f"{i + 1}. Custom hash function with {Hash.k_plot[i]}\n with a max SE of {np.log2(self.p):.3f}"
        else:
            return f"{i + 1}. Hash Analysis for key {Hash.key_plot[i]}"