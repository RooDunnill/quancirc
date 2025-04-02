import numpy as np

def matrix_fwht(rho):
    """Applies the Fast Walsh Hadamard Transform to a density matrix rho."""
    sqrt2_inv = 1 / np.sqrt(2)
    dim = rho.shape[0] 
    

    for i in range(int(np.log2(dim))):
        step_size = 2 ** (i + 1)  
        half_step = step_size // 2  
        
    
        outer_range = np.arange(0, dim, step_size)[:, None]
        inner_range = np.arange(half_step)
        indices = outer_range + inner_range
        
        
        a, b = rho[indices, :], rho[indices + half_step, :]
        rho[indices, :] = (a + b) * sqrt2_inv
        rho[indices + half_step, :] = (a - b) * sqrt2_inv

    
    for i in range(int(np.log2(dim))):  
        step_size = 2 ** (i + 1) 
        half_step = step_size // 2  
        
        
        outer_range = np.arange(0, dim, step_size)[:, None]
        inner_range = np.arange(half_step)
        indices = outer_range + inner_range
        

        a, b = rho[:,indices], rho[:,indices + half_step]
        rho[:,indices] = (a + b) * sqrt2_inv
        rho[:,indices + half_step] = (a - b) * sqrt2_inv
    
    return rho