Welcome to my program, quancirc. Thie simulator was mostly built as a way for me to improve my programming skills and also learn more
quantum information techniques and applications. The goal of this project is to not only implement a circuit that can simulate realistically
how quantum computers (atleast the ones we strive for, ie have all the gates) run, but also to create various other programs within, such as running
algorithms efficiently on the circuit, and also to create quantum cryptography protocols such as BB84 or QCF.


Installation Process:  
first from outside the program, run pip install -e quancirc  
then to run any of the scripts use:  
python -m quancirc.src.scripts."script_name"  


Honestly my code changes too much atm to give a detailed indepth guide to each segment so heres a brief overview of the whole program:  
General Circuit: Allows for general manipulations of Qubits, has a circuit and multi state processes on that circuit, ie can run two circuits simultaneously, need to update this to allow for quant info comparisons of various circuits
Lightweight Circuit: This is a pure state only, stripped down version designed originally for fast grover search simulation and honeslty thats all I use it for now, is quite outdated but has niche uses
Symbolic Circuit: I created the start of this when I was looking into plotting the cheating probabilities of a few QCF protocols, will hopefully flesh it out more during the summer when I work more on my diss and need some analysis tools
Qutrit Circuit: CUrrently what I am working on, and is almost identical to the General Qubit circuit but with Qutrits, has a lot less predefined states as in su(3) rather than su(2) but has so many more gates that I will need to implement
Crypto Protocols: This is a little side project, where I try to simulate protocols on the quantum computer, so far have done BB84 and am working on BBM92

I have many example files for you to check out as a showcase or as a tutorial, although I don't regularly update all the newest featurs in there till I have them really set up well