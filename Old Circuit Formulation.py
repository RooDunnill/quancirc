class Circuit:
    def __init__(self, name, matrix_array):
        self.name = name
        self.matrix_array = np.array(matrix_array,dtype=np.complex128)

    def __str__(self):
        return f"{self.name}({self.matrix_array})"
    

    def gate_mult(self):
        new_name = f"{self.name}"
        num_mats = len(self.matrix_array)
        matrix_length = int(sum(len(p) for p in self.matrix_array)/num_mats)
        dim = int(sp.sqrt(matrix_length))
        print(dim)
        new_mat = np.zeros(int(matrix_length),dtype=np.complex128)
        final_mat = np.zeros(int(matrix_length),dtype=np.complex128)
        summ = np.zeros(1,dtype=np.complex128)
        for m in range(num_mats-1):
            for i in range(dim):
                for k in range(dim):
                    for j in range(dim):
                        summ[0] += (self.matrix_array[m,j+dim*i]*self.matrix_array[m+1,k+j*dim])
                    new_mat[k+dim*i] += summ[0]
                    summ = np.zeros(1,dtype=np.complex128)
            for t in range(matrix_length):
                self.matrix_array[m+1,t]=new_mat[t]
            new_mat = np.zeros(int(matrix_length),dtype=np.complex128)
        print(self.matrix_array)
        for h in range(matrix_length):
            final_mat[h] = self.matrix_array[m+1,h]
        return Circuit(new_name, np.array(final_mat))
oneqcircuit_mat_array = [X_Gate.matrix,Hadamard.matrix,X_Gate.matrix]
oneqcircuit = Circuit("1 Qubit Circuit", oneqcircuit_mat_array)
print(oneqcircuit.gate_mult())