import matplotlib.pyplot as plt
import numpy as np
from ...circuit.classes.circuit import *
from ...circuit.classes.quant_info import *
from ...circuit.classes.qubit import *
from ...circuit.classes.gate import *
from matplotlib.widgets import Slider

phi = np.linspace(0, np.pi/4, 1000)
fidelity = np.zeros(1000)
detection_fidelity = np.zeros(1000)
state_1 = Qubit(state=[1,0])
alpha = 0

def update(alpha_val):
    for i, angle in enumerate(phi):
        fidelity[i] = QuantInfo.fidelity(state_1, Gate.Rotation_X(angle + alpha_val) @ state_1)
        detection_fidelity[i] = QuantInfo.fidelity(state_1, Gate.Rotation_X(angle) @ state_1)

    det_prob = ((1 - detection_fidelity) * np.sin(alpha_val)**2)/2
    prob_a = 1/2 + np.sqrt(fidelity) * np.sin(2 * (alpha_val))/2 - det_prob
    prob_b = 1/2 + np.abs(np.cos(2* phi)/2)

    ax.clear()
    ax.plot(phi, prob_a, color="red", label=f"Pr[Alice Wins if cheating]")
    ax.plot(phi, prob_b, color="black", label=f"Pr[Bob Wins if cheating]")
    ax.plot(phi, fidelity, color="green", label=f"Fidelity between states")
    ax.plot(phi, det_prob, color="blue", label=f"Pr[Bob detects Alice cheating]")
    ax.plot(phi, np.abs(prob_a - prob_b))
    ax.axvline(x=np.pi/8, color='g', linestyle='--', label=r"$\phi = \frac{\pi}{8}$")

    ax.set_xlabel(f"$\phi$ radians")
    ax.set_ylabel(f"Trace Distance")
    ax.set_title(f"Trace distance against angle $\phi$ between two states")
    ax.legend()
    ax.grid(True)
    plt.draw()


fig, ax = plt.subplots(figsize=(10,8))

update(alpha)

ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])  
slider = Slider(ax_slider, 'Alpha', 0, np.pi/4, valinit=alpha, valstep=np.pi/128)


slider.on_changed(update)

plt.show()
