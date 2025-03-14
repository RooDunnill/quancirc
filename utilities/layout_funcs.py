import numpy as np


def top_probs(prob_list: np.ndarray, **kwargs) -> np.ndarray:             #sorts through the probability distribution and finds the top n probabilities corresponding to the length n or the oracle values
        """Computes the top n probabilities of a list of probabilities"""
        n = kwargs.get("n", 8)
        non_zero_count = np.count_nonzero(prob_list)
        n = non_zero_count if non_zero_count <= n else n              #uses less than the given input if there are only a few non zero values
        top_n = np.array([], dtype=prob_list.dtype)
        temp_lst = prob_list.copy()  
        for _ in range(n):
            max_value = np.max(temp_lst)
            top_n = np.append(top_n, max_value)
            temp_lst = np.delete(temp_lst, np.argmax(temp_lst))
        result = []
        used_count = {} 
        for i, num in enumerate(prob_list):
            if num in top_n and used_count.get(num, 0) < np.count_nonzero(top_n == num):        #this accounts for if you have two numbers with the same value
                result.append((i, num))
                used_count[num] = used_count.get(num, 0) + 1
        return np.array(result, dtype=object)

def format_ket_notation(list_probs: np.ndarray, **kwargs) -> str:
    """Used for printing out as it gives each state and the ket associated with it"""
    list_type = kwargs.get("type", "all")
    num_bits = kwargs.get("num_bits", int(np.ceil(np.log2(len(list_probs)))))    #this is to flush out the ket notation to give the correct number of bits back
    prec = kwargs.get("precision", 3)
    if list_type == "topn":
        print_out = f""
        for ket_val, prob_val in zip(list_probs[:,0],list_probs[:,1]):       #iterates through every prob value
            print_out += (f"State |{bin(ket_val)[2:].zfill(num_bits)}> ({ket_val}) with a prob val of: {prob_val * 100:.{prec}f}%\n")
        return print_out
    elif list_type == "all":         #this just does it for every single ket value
        ket_mat = range(len(list_probs))
        print_out = f""
        for ket_val, prob_val in zip(ket_mat,list_probs):
            print_out += (f"|{bin(ket_val)[2:].zfill(num_bits)}>  {prob_val * 100:.{prec}f}%\n")
        return print_out