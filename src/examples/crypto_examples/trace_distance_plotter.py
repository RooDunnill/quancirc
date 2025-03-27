import matplotlib.pyplot as plt
import numpy as np
from ...circuit.classes.circuit import *
from ...circuit.classes.quant_info import *
from ...circuit.classes.qubit import *
from ...circuit.classes.gate import *

phi = np.linspace(0, np.pi/2, 1000)
fidelity = np.zeros(1000)
prob_b = 1/2 + np.abs(np.cos(2* phi)/2)
for i, angle in enumerate(phi):
    fidelity[i] = QuantInfo.fidelity(q0, Gate.Rotation_Z(angle) @ q0)

prob_a = 1/2 + np.sqrt(fidelity) * (np.sin(2 * phi))/2

plt.figure(figsize=(8,6))
plt.plot(phi, prob_a, color="red", label=f"Pr[Alice Wins if cheating]")
plt.plot(phi, prob_b, color="black", label=f"Pr[Bob Wins if cheating]")
plt.axvline(x=np.pi/8, color='g', linestyle='--', label=r"$\phi = \frac{\pi}{8}$")
plt.xlabel(f"$\phi$ radians")
plt.ylabel(f"Trace Distance")
plt.title(f"Trace distance against angle $\phi$ between two states")
plt.legend()
plt.grid()
plt.show()
