from QC_Functions import *
oracle_values = [2, 4, 8, 16]
def grover_alg(oracle_values, n, iterations=None):
    console.rule(f"Grovers algorithm with values: {oracle_values}", style="headers")
    op_iter = (np.pi/4)*np.sqrt((2**n)/len(oracle_values)) - 0.5
    if iterations == None:
        iterations = op_iter
        if iterations < 1:
            iterations = 1
        print_array(f"Optimal amount of iterations are: {iterations}")
    qub = q0
    had_op = Hadamard
    flip_cond = - np.ones(qub.dim**n)
    flip_cond[0] += 2
    for i in range(n-1):
        qub **= q0
        had_op **= Hadamard
    it = 0
    while it < int(iterations):
        if it != 0:
            qub = final_state
        initialized_qubit = had_op * qub
        intermidary_qubit = had_op * phase_oracle(initialized_qubit, oracle_values)
        for j, vals in enumerate(flip_cond):
            intermidary_qubit.vector[j] = intermidary_qubit.vector[j] * vals 
        final_state = had_op * intermidary_qubit
        it += 1
        print_array(f"Iteration number: {it}")
        final_state.name = f"Grover Search with Oracle Values {oracle_values}, after {int(iterations)} iterations is: "
    final_state = final_state.prob_dist()
    return final_state, op_iter
print_array(grover_alg(oracle_values,7)[0])