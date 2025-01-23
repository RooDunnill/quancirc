# QC-Simulator
TODO: FIX MEASUREMENT FUNCTION, (i believe it is fixed?)
MIXED STATE DENSITY MATRIX (NOT ENOUGH KNOWLEDGE YET)
ENTANGLEMENT VARIABLE? AFTER CNOT GATES
MEASUREMENT FUNCTION
A CIRCUIT COMPILER OF SOME SORT
Hi and welcome to my Quantum Computing Simulator! I am currently a masters student studying Theoretical Physics but have a passion for quantum computing and wanted to grasp a more fundamental understandings of quantum circuitry so made this. Also it helped me practice coding. Any feedback would be great! Please find a list of instructions below. P.S. This is still a work in progress

Classes:
timer - used mostly as debugging to see the time elapsed to improve performance
Qubit - i have made 4 pre-existing qubits q0,q1,q+,q-
        Functions:
            Tensor Product:used with the @ symbol between two qubit types
            Normaliser: .norm() to normalise a qubit, mostly used for making custom qubits
            Qubit Info: .qubit_infov() just background info
            Measure: .measure(qubit_number) measures a specific qubit value
            Density Matrix: .density_mat() computes the density matrix for a given qubit or multi-qubit state TODO: make for mixed states of qubits (maybe with addition function of some kind)
            Bloch Sphere Plot: bloch_plot() can be used to plot as many qubits on the bloch sphere as desired TODO: again make superposition states
            Probability of given state: .prob_state(measuring state, optional gate), will give you the probability for a given state of qubits
            Probability of all states: .prob_dist(optional gate), will output all the probabilities for all the possible states
            
Gate - i have coded in most of the main gates, however the option to make your own is available.
        Current Gates:
            Pauli X Gate: Flips the qubit from |+> to |-> and vice versa. Commonly used in the CNot gate, which is made from this and a control gate.
            Pauli Y Gate: Flips the qubit from |i> to |-i> and vice versa.
            Pauli Z Gate: Flips the qubit in the computational basis |1> and |0>.
            Identity: Standard 2 x 2 identity matrix. Mostly used for building CNot gates.
            Hadamard: flips the qubit between the computational basis and the {|+>,|->} basis and vice versa. |0> to |+> and |1> to |->.
            U_Gates: This gate can be made into most gates through the implementation of 3 variables. Basically is a custom gate but you can also write your own gate in easilly.
            C_Gates: This gate is used in many different ways to entangle qubits and to make decisions on one qubits based on the state of another. I have made a function to make a custom one.
            TODO: make the toffoli gate, time evolution gates and 1/2 and 1/4 gates
        Functions:
            Tensor Product: Denoted by @, used to combine two gates, density matrices or a mix. Makes a matrix that has the dimensions of the product of their Hilbert spaces.
            Matrix Multiplication: Denoted by * and can be done between gates, density matrics and qubits, but not between two qubits.
            Direct Sum: Denoted by +, it is mostly used to make up the custom CNot gates.
            Equal Plus: Denoted by +=, this allows the direct sum of gates but with this notation. Again used mostly for CNot gate production.
            Gate Info: .gate_info, just gives info on gates and what they do.
Control Gate: While technically not a function, you can create a C Gate and pick the gate you wish to implement in it, along with in which "qubits" you wish to implement these. e.g. 
C_Gate("CNot", qc_dat.C_Not_matrix, X_Gate, 1, 2), would create a CNot gate acting on qubit 2 from qubit 1. It must always act on qubit 1 in either direction, but that is more a notation convention and when creating a circuit you can choose which qubit to act it on.
U_Gate: Contains its own class so that you can implement its three values a,b and c.
Density: Used for the density matrices, mostly defined as its own class for isinstance() functions.
Prob_dist: Is made its own class to allow it to have its own printing format.
Print_array: Used to print the matrices in a more coherrent manner.
QC_error: Used to make custom error messages
qc_dat: Used to store lots of strings and pre-existing matrix values.


print_array
