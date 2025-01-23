# QC-Simulator
TODO: FIX MEASUREMENT FUNCTION (TENSOR PRODUCTS NOT DIRECT SUMS)
N X N DENSITY MATRIX
N X N MEASUREMENT OF MULTIPLE QUBITS
ENTANGLEMENT VARIABLE? AFTER CNOT GATES
Hi and welcome to my Quantum Computing Simulator! I am currently a masters student studying Theoretical Physics but have a passion for quantum computing and wanted to grasp a more fundamental understandings of quantum circuitry so made this. Also it helped me practice coding. Any feedback would be great! Please find a list of instructions below. P.S. This is still a work in progress

Classes:
timer - used mostly as debugging to see the time elapsed to improve performance
gate_data - used to store mostly pre-existing matrix values and long strings of information
Qubits - i have made 4 pre-existing qubits q0,q1,q+,q-
        Functions:
            Tensor Product:used with the @ symbol between two qubit types
            Normaliser: .norm() to normalise a qubit, mostly used for making custom qubits
            Qubit Info: .qubit() just background info
            Measure: .measure(qubit_number) measures a specific qubit value
            Density Matrix: .density_mat() computes the density matrix for a given qubit TODO: make for n qubits or superposition of qubits
            Bloch Sphere Plot: bloch_plot() can be used to plot as many qubits on the bloch sphere as desired TODO: again make superposition states
            
Gates
U_Gates
C_Gates
print_array
