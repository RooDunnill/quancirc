import matplotlib.pyplot as plt
import numpy as np
from ...circuits.base_circuit import *
from ...circuits.symbolic_circuit import *
from matplotlib.widgets import Slider




def aharonov_plotter_example():
    phi = np.linspace(0, np.pi/2, 200)
    symb_fidelity = np.zeros(200)
    symb_trace_distance = np.zeros(200)
    phi_symb = sp.symbols("phi_symb")
    alpha = 0
    for i, angle in enumerate(phi):
        symb_state_0 = SymbQubit(rho=[[sp.cos(phi_symb)**2, 0],[0, sp.sin(phi_symb)**2]])
        symb_state_1 = SymbQubit(rho=[[sp.sin(phi_symb)**2, 0],[0, sp.cos(phi_symb)**2]])
        symb_state_0.rho = symb_state_0.rho.subs({sp.symbols("phi_symb"):angle})
        symb_state_1.rho = symb_state_1.rho.subs({sp.symbols("phi_symb"):angle})
        symb_trace_distance[i] = SymbQuantInfo.trace_distance(symb_state_0, symb_state_1).evalf()
    def update(alpha_val):
        for i, angle in enumerate(phi):
            symb_state_0 = SymbQubit(rho=[[sp.cos(phi_symb)**2, 0],[0, sp.sin(phi_symb)**2]])
            symb_state_1 = SymbQubit(rho=[[sp.sin(phi_symb)**2, 0],[0, sp.cos(phi_symb)**2]])
            
            symb_state_0.rho = symb_state_0.rho.subs({sp.symbols("phi_symb"):angle + alpha_val})
            symb_state_1.rho = symb_state_1.rho.subs({sp.symbols("phi_symb"):angle + alpha_val})
            symb_fidelity[i] = SymbQuantInfo.fidelity(symb_state_0, symb_state_1).evalf()
            
            
        prob_a = 1/2 + np.sqrt(symb_fidelity) * np.sin(2 * (alpha_val))/2
        prob_b = 1/2 + np.abs(symb_trace_distance/2)
        
        ax.clear()
        ax.plot(phi, prob_a, color="red", label=f"Pr[Alice Wins if cheating]")
        ax.plot(phi, prob_b, color="black", label=f"Pr[Bob Wins if cheating]")
        ax.plot(phi, symb_fidelity, color="red", linestyle="--", label=r"Fidelity")
        ax.plot(phi, symb_trace_distance, color="black", linestyle="--", label="trace distance")
        ax.axvline(x=np.pi/8, color='green', linestyle=':', label=r"$\phi = \frac{\pi}{8}$")

        ax.set_xlabel(r"$\phi$ radians")
        ax.set_ylabel(f"Probabilities")
        ax.set_title(f"Alice and Bob's cheating probabilities")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.00), fontsize=10, frameon=False)
        ax.text(1.7, 0.7, r"$Pr_A = \frac{1}{2} + \sqrt{F(\rho_0, \rho_1)} \sin(2\alpha) / 2$", fontsize=12, color="red", bbox=dict(facecolor='white', alpha=0.5))
        ax.text(1.7, 0.6, r"$Pr_B = \frac{1}{2} + \frac{|T(\rho_0, \rho_1)|}{2}$", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.5))
        ax.text(1.7, 0.5, r"$F(\rho_0, \rho_1) = \left(\text{Tr} \sqrt{\sqrt{\rho_0} \rho_1 \sqrt{\rho_0}}\right)^2$", fontsize=12, color="red", bbox=dict(facecolor='white', alpha=0.5))
        ax.text(1.7, 0.4, r"$T(\rho_0, \rho_1) = \frac{1}{2} \text{Tr} |\rho_0 - \rho_1|$", fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.5))
        ax.set_xlim([0, np.pi/2])
        ax.set_ylim([0, 1])
        ax.grid(True)
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/np.pi:.1f}π'))
        ax.set_xticks(np.linspace(0, np.pi/2, 10))
        fig.subplots_adjust(right=0.75)
        plt.tight_layout(pad=2.0)
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(10,8))
    
    update(alpha)
    ax.set_xlim([0, np.pi/2])
    ax.set_ylim([0, 1])
    ax.grid(True)
    fig.subplots_adjust(right=0.75)
    plt.tight_layout(pad=2.0)
    ax.set_facecolor("#f4f4f4")

    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])  
    slider = Slider(ax_slider, "alpha", 0, np.pi/4, valinit=alpha, valstep=np.pi/32)

    slider.on_changed(update)

    slider.ax.set_facecolor("lightgray")
    slider.valtext.set_fontsize(12)
    
    plt.show()

if __name__ == "__main__":
    aharonov_plotter_example()
