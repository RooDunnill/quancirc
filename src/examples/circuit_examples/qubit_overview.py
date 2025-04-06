import numpy as np
from ...circuits.general_circuit.classes.qubit import *
from ...circuits.general_circuit.classes.gate import Hadamard
from ...config.config import linewid

def qubit_overview():
    print(f"Welcome to the Qubit Overview".center(linewid, "^"))
    print("\n")
    print(f"Instantiating Qubits".center(linewid, "-"))
    print(f"In this tutorial, I will go through all you can do with the base qubits")
    print(f"This will include purely the QUbit class and not its interactions with other classes")
    print(f"So first, how we instantiate a Qubit, this can be done in a custom manner or through the use of predefined qubits")
    print(f"You can instantiate one as such:")
    demo_qubit_1 = Qubit(rho=[[1,0],[0,0]])
    print(demo_qubit_1)
    print(f"Or you can instantiate the same Qubit as such:")
    demo_qubit1_1 = Qubit(state=[1,0])
    print(demo_qubit1_1)
    print(f"Or for ease of use, for the six most common QUbits, you can just do this:")
    demo_qubit_1_2 = q0
    print(demo_qubit_1_2)
    print(f"As you can see these are all the same!")
    print(f"To combine qubits, to create multi-qubit systems, you can use the '%' symbol that denotes the tensor product")
    demo_qubit_2 = q0 % q0
    print(demo_qubit_2)
    print(f"As you can see, we have simply made a two qubit state, as for many circuit, the base state is |000...>, I added the feature to allow for you to")
    print(f"easilly instantiate as many as you would like as such:")
    demo_qubit_2_2 = Qubit.q0(n=2)
    print(demo_qubit_2_2)
    print(f"As you can see, this once again does the same thing")
    print(f"\n")

    print(f"Display Mode".center(linewid, "-"))
    print(f"Quantum states can be represented in different ways, using the method set_display_mode, you can change the display of the Qubit class")
    demo_display_qub = q0 % qp
    print(demo_display_qub)
    print(f"The default is in 'density' mode, where only the density is displayed")
    demo_display_qub.set_display_mode("vector")
    print(f"If we wanted just the vector of probs in the computational basis, we can set it as seen to get:")
    print(demo_display_qub)
    demo_display_qub.set_display_mode("both")
    print(f"And for max detail we can set it as seen to obtain:")
    print(demo_display_qub)


    print(f"Matrix Multiplication".center(linewid, "-"))
    print(f"While not necessarily used within a circuit, matrix multiplication is frequently used within Quantum Information calculations such as fidelity,")
    print(f"As such, I have encorporated that feature as well which is simple done as such:")
    demo_qubit_3_1 = qm % qm
    demo_qubit_3_2 = qp % qp
    print(demo_qubit_3_1 @ demo_qubit_3_2)
    print(f"Ironically, this example creates a completely empty array, which is why the internal validation is skipped for matrix multiplication of qubits")
    print(f"You can also see that when they are printed, it gives the type of the Quantum State, this here is Non-Unitary as the state does not satisfy the")
    print(f"standard conditions to be a rho matrix.")
    print(f"Briefly, this is naturally how you would apply gates to the Qubits, or how it is done within the circuit:")
    print(Hadamard @ q0)
    print("\n")


    print(f"Other Operations".center(linewid, "-"))
    print(f"Lets briefly look at the other basic operations you can do, again nearly solely used for Quantum Information as they do not have a physical rep")
    print(f"For these examples I will use two 2-qubit systemes:")
    demo_qubit_4_1 = q1 % qp
    demo_qubit_4_2 = qm % q0
    print(demo_qubit_4_1)
    print(demo_qubit_4_2)
    print(f"Subtraction:")
    print(demo_qubit_4_1 - demo_qubit_4_2)
    print(f"Addition:")
    print(demo_qubit_4_1 + demo_qubit_4_2)
    print(f"Division:")
    print(demo_qubit_4_1 / 2)
    print(f"Multiplication:")
    print(demo_qubit_4_2 * 2)
    print(f"Direct Sum:")
    print(demo_qubit_4_1 & demo_qubit_4_2)
    print(f"Sum examples of uses are subtraction for Trace Distance calculations or multiplication for Mixed state creation")

    print(f"Creating Mixed States".center(linewid, "-"))
    print(f"There are several ways to make mixed states, the first way is by defining a qubit instance as mixed:")
    demo_qubit_5_1 = Qubit(state=[[1,0],[0,1]], weights=[0.5,0.5])
    print(demo_qubit_5_1)
    print(f"As you can see, this is a mix of a |0> and |1> state with half probability of each state, not to be confused with a |+> state")
    print(f"By having most state representations in 'density' format, it helps to really highlight the difference between the two")
    print(f"An easier way to define states, especially for more complex systems is as such:")
    demo_qubit_5_2 = Qubit.create_mixed_state([qp, qm, q0, q1], [0.2,0.3,0.4,0.1])
    print(demo_qubit_5_2)
    print(f"This allows for total control over your mixed states and can be more easilly scaled as you can simply do something like this:")
    print(Qubit.create_mixed_state([demo_qubit_4_1, demo_qubit_4_2], [0.5,0.5]))
    print(f"To create multi-qubit systems from pre-existing states")
    print(f"As you can also see in the print, the type 'Mixed' is shown when these are printed too, and if say by using a channel, a state goes from pure to mixed,")
    print(f"That will be updated accordingly. Mixed states can only be printed in 'density' mode")

    print(f"Indexing and isolating Qubits".center(linewid, "-"))
    print(f"When we have large quantum states, sometimes we would like to look at individual qubits")
    print(f"This can be acheived successfully with partial tracing, which is the core premise of the functions behind this section")
    demo_qubit_6 = qp % qm % q0
    print(f"Lets take this 3-qubit system:")
    print(demo_qubit_6)
    print(f"I made each qubit different to highlight the process more, using indexing and overriding the dunder methods __setitem__ and __getitem__, we can do this:")
    print(demo_qubit_6[0])
    print(demo_qubit_6[1])
    print(demo_qubit_6[2])
    print(f"This partially traces out all of the qubits bar the one you wish, we can also do this for slice indexing to return multiple qubits:")
    print(demo_qubit_6[1:3])
    print(f"As you can see, this returned qubits 1,2")
    print(f"We can also set qubtis as new qubits here, to simulate gates being acted on or other processes:")
    demo_qubit_6[2] = Hadamard @ demo_qubit_6[2]
    print(demo_qubit_6[2])
    print(f"As you can see, we applied the Hadamard gate to just one of the qubits and then put it back into the state")
    print(f"Now if we print the whole state, we will see that it has changed:")
    print(demo_qubit_6)

    print(f"General Information".center(linewid, "-"))
    print(f"If you ever need debugging information for your quantum state, you can call the method debug() to display all the attributes of the state and other useful information")
    demo_qubit_7 = Qubit.create_mixed_state([q0 % q0, q1 % q0, qpi % qm], [0.3,0.2,0.5])
    demo_qubit_7.debug()
    print(f"As you can see, this breaks down each qubit and displays as much information as possible")
    print(f"As you may have picked up on, every qubit has an ID value assocaited with it, for premade qubits they are as such:")
    print(f"q0 ID value: {q0.id}")
    print(f"While for self-made quantum states, they are simply numbered from 0 upwards. Along with each ID however, they also have their own history associated with them")
    print(f"Lets take a quantum state such as the one below that has been through a few operations:")
    demo_qubit_7_2 = q0 % q1
    demo_qubit_7_2 @= (qp % qm)
    demo_qubit_7_2 *= 2
    demo_qubit_7_2 = (Hadamard % Hadamard) @ demo_qubit_7_2
    print(f"We can display its history by calling  the print_history() method:")
    demo_qubit_7_2.print_history()
    print(f"This is a great way to debug, to just differentiate between two states or to recap all of your operations!")

if __name__ in "__main__()":
    qubit_overview()