import matplotlib.pyplot as plt
import numpy as np
from ...circuit.classes.circuit import *
from ...circuit.classes.quant_info import *
from ...circuit.classes.qubit import *
from ...circuit.classes.gate import *
from matplotlib.widgets import Slider






def aharonov_plotter_example():
    phi = np.linspace(0, np.pi/2, 200)
    fidelity = np.zeros(200)
    detection_fidelity = np.zeros(200)
    trace_bound_upper = np.zeros(200)
    trace_bound_lower = np.zeros(200)
    state_1 = Qubit(state=[1,0])
    alpha = 0
    detection_fidelity[:] = [QuantInfo.fidelity(state_1, Gate.Rotation_X(angle) @ state_1) for angle in phi]

    def update(alpha_val):
        fidelity[:] = [QuantInfo.fidelity(state_1, Gate.Rotation_X(angle + alpha_val) @ state_1) for angle in phi]
        trace_bound_upper[:] = [QuantInfo.trace_distance_bound(state_1, Gate.Rotation_X(angle + alpha_val) @ state_1)[0] for angle in phi]
        trace_bound_lower[:] = [QuantInfo.trace_distance_bound(state_1, Gate.Rotation_X(angle + alpha_val) @ state_1)[1] for angle in phi]
        det_prob = ((1 - detection_fidelity) * np.sin(alpha_val)**2)/2
        prob_a = 1/2 + np.sqrt(fidelity) * np.sin(2 * (alpha_val))/2 - det_prob
        prob_b = 1/2 + np.abs(np.cos(2* phi)/2)
        
        ax.clear()
        ax.plot(phi, prob_a, color="red", label=f"Pr[Alice Wins if cheating]")
        ax.plot(phi, prob_b, color="black", label=f"Pr[Bob Wins if cheating]")
        ax.axvline(x=np.pi/8, color='g', linestyle='--', label=r"$\phi = \frac{\pi}{8}$")
        ax.plot(phi, fidelity, color="green", label=f"Fidelity between states")
        ax.plot(phi, det_prob, color="blue", label=f"Pr[Bob detects Alice cheating]")
        ax.plot(phi, trace_bound_upper)
        ax.plot(phi, trace_bound_lower)
        ax.set_xlabel(r"$\phi$ radians")
        ax.set_ylabel(f"Probabilities")
        ax.set_title(f"Alice and Bob's cheating probabilities")
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10, frameon=False)
        ax.grid(True)
        ax.set_facecolor("#f4f4f4")
        
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(10,8))
    update(alpha)


    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])  
    slider = Slider(ax_slider, "alpha", 0, np.pi/4, valinit=alpha, valstep=np.pi/32)

    slider.on_changed(update)

    slider.ax.set_facecolor("lightgray")
    slider.valtext.set_fontsize(12)

    plt.show()

if __name__ == "__main__":
    aharonov_plotter_example
