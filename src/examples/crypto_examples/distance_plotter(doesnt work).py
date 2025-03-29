import matplotlib.pyplot as plt
import numpy as np
from ...circuits.base_circuit import *
from ...circuits.symbolic_circuit.classes import *
from matplotlib.widgets import Slider
import sympy as sp
import copy


def aharonov_plotter_example():
    phi = np.linspace(0, np.pi/2, 200)
    symb_fidelity = np.zeros(200)
    symb_trace_distance = np.zeros(200)
    phi_symb = sp.symbols("phi_symb")
    alpha = 0
    beta = 0

    cos_phi = sp.cos(phi_symb)
    sin_phi = sp.sin(phi_symb)

    state_0_0 = SymbQubit(state=[cos_phi, sin_phi])
    state_1_0 = SymbQubit(state=[sin_phi, -sin_phi])
    state_0_1 = SymbQubit(state=[sin_phi, -cos_phi])
    state_1_1 = SymbQubit(state=[sin_phi, cos_phi])

    input_state_0 = SymbQubit.create_mixed_state([state_0_0, state_1_0], [0.5,0.5])
    input_state_1 = SymbQubit.create_mixed_state([state_0_1, state_1_1], [0.5,0.5])

    rho_0_expr = input_state_0.rho
    rho_1_expr = input_state_1.rho

    rho_0_values = [rho_0_expr.subs(phi_symb, angle).evalf() for angle in phi]
    rho_1_values = [rho_1_expr.subs(phi_symb, angle).evalf() for angle in phi]

    symb_trace_distance = np.array([SymbQuantInfo.trace_distance(SymbQubit(rho=rho_0), SymbQubit(rho=rho_1)).evalf() for rho_0, rho_1 in zip(rho_0_values, rho_1_values)])

    def update(alpha_val, beta_val):
        rho_0_values = [rho_0_expr.subs(phi_symb, angle + alpha_val).evalf() for angle in phi]
        rho_1_values = [rho_1_expr.subs(phi_symb, angle + alpha_val).evalf() for angle in phi]

        symb_fidelity[:] = [SymbQuantInfo.fidelity(SymbQubit(rho=rho_0), SymbQubit(rho=rho_1)).evalf() for rho_0, rho_1 in zip(rho_0_values, rho_1_values)]

        prob_a = 1/2 + np.sqrt(symb_fidelity) * np.sin(2 * (alpha_val))/2
        prob_b = 1/2 + np.abs(symb_trace_distance/2) * beta_val

        ax.clear()
        ax.plot(phi, prob_a, color="red", label=f"Pr[Alice Wins if cheating]")
        ax.plot(phi, prob_b, color="black", label=f"Pr[Bob Wins if cheating]")
        ax.plot(phi, symb_fidelity, color="red", linestyle="--", label=r"Fidelity")
        if beta_val == 1:
            ax.plot(phi, symb_trace_distance, color="black", linestyle="--", label="trace distance")
        ax.axvline(x=np.pi/8, color='green', linestyle=':', label=r"$\phi = \frac{\pi}{8}$")

        ax.set_xlabel(r"$\phi$ radians")
        ax.set_ylabel(f"Probabilities")
        ax.set_title(f"Alice and Bob's cheating probabilities")
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.2), fontsize=10, frameon=False)
        ax.text(1.7, 0.6, r"$Pr_A = \frac{1}{2} + \sqrt{F(\rho_0, \rho_1)} \sin(2\alpha) / 2$", fontsize=12, color="red", bbox=dict(facecolor='white', alpha=0.5))
        ax.text(1.7, 0.4, r"$Pr_B = \frac{1}{2} + \frac{|T(\rho_0, \rho_1)|}{2}$", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.5))
        ax.text(1.7, 0.2, r"$F(\rho_0, \rho_1) = \left(\text{Tr} \sqrt{\sqrt{\rho_0} \rho_1 \sqrt{\rho_0}}\right)^2$", fontsize=12, color="red", bbox=dict(facecolor='white', alpha=0.5))
        ax.text(1.7, 0.0, r"$T(\rho_0, \rho_1) = \frac{1}{2} \text{Tr} |\rho_0 - \rho_1|$", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.5))
        ax.set_xlim([0, np.pi/2])
        ax.set_ylim([0, 1])
        ax.grid(True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/np.pi:.1f}π'))
        ax.set_xticks(np.linspace(0, np.pi/2, 6))
        ax.set_yticks(np.linspace(0,1,5))
        fig.subplots_adjust(right=0.5, bottom=0.1)
        plt.tight_layout(pad=3.0)
        fig.canvas.draw_idle()

    fig = plt.figure(figsize=(10,8))
    grid = fig.add_gridspec(3, 1, height_ratios=[5, 0.25, 0.25])
    ax = fig.add_subplot(grid[0])
    update(alpha, beta)
    ax.set_xlim([0, np.pi/2])
    ax.set_ylim([0, 1])
    ax.grid(True)
 
    ax.set_facecolor("#f4f4f4")

    ax_slider_a = fig.add_subplot(grid[1], position=[0.1, 0.5, 0.8, 0.5])
    ax_slider_a.set_yticks([])
    ax_slider_a.set_yticklabels([])
    slider_a = Slider(ax_slider_a, "alpha", 0, np.pi/4, valinit=alpha, valstep=np.pi/64, color="red")

    ax_slider_b = fig.add_subplot(grid[2], position=[0.1, 0.5, 0.8, 0.5])
    ax_slider_b.set_yticks([])
    ax_slider_b.set_yticklabels([])
    slider_b = Slider(ax_slider_b, "beta", 0, 1, valinit=beta, valstep=1, color="black")

    slider_a.on_changed(lambda val: update(val, slider_b.val))
    slider_b.on_changed(lambda val: update(slider_a.val, val))

    slider_a.ax.set_facecolor("lightgray")
    slider_a.valtext.set_fontsize(12)

    slider_b.ax.set_facecolor("lightgray")
    slider_b.valtext.set_fontsize(12)
    slider_a.valtext.set_color('red')
    slider_b.valtext.set_color('black')
    plt.show()

if __name__ == "__main__":
    aharonov_plotter_example()