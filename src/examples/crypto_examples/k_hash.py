from ...crypto_protocols.primitives import *
import numpy as np
import matplotlib.pyplot as plt

def k_hash_example():
    list_vals = np.array(range(10000))
    k = 2
    p = 3
    rand_key = n_length_int_key(24)
    k_2_hash = Hash(k=k, key=rand_key, p=p)
    k_2_hash.analyse(k_2_hash.k_hash(list_vals))

    k_3_hash = Hash(k=k+1, key=rand_key, p=p)
    k_3_hash.analyse(k_3_hash.k_hash(list_vals))

    k_4_hash = Hash(k=k+2, key=rand_key, p=p)
    k_4_hash.analyse(k_4_hash.k_hash(list_vals))

    k_6_hash = Hash(k=k+4, key=rand_key, p=p)
    k_6_hash.analyse(k_6_hash.k_hash(list_vals))

    k_8_hash = Hash(k=k+6, key=rand_key, p=p)
    k_8_hash.analyse(k_8_hash.k_hash(list_vals))

    k_12_hash = Hash(k=k+10, key=rand_key, p=p)
    k_12_hash.analyse(k_12_hash.k_hash(list_vals))
    plt.show()

if __name__ == "__main__":
    k_hash_example()
    