import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def delta_sinc(x, L):
    """Fourier-style approximation using sinc structure, more QFT based"""
    result = np.zeros_like(x)
    nonzero = x != 0
    result[nonzero] = np.sin(L * x[nonzero]) / (np.pi * x[nonzero])
    result[~nonzero] = L / np.pi
    return result


def delta_gaussian(x, epsilon):
    """Gaussian approximation to delta function, more simple visualisation than QFT approach"""
    return (1 / (np.sqrt(2 * np.pi) * epsilon)) * np.exp(-x**2 / (2 * epsilon**2))


x = np.linspace(-0.02, 0.02, 10000)
initial_L = 1

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

line_sinc, = ax.plot(x, delta_sinc(x, initial_L), lw=2, label='sinc')
line_gauss, = ax.plot(x, delta_gaussian(x, 1 / initial_L), lw=2, label='Gaussian')
ax.set_title(r"Dirac delta approximations: $\frac{\sin(Lx)}{\pi x}$ vs Gaussian")
ax.set_xlabel("x")
ax.set_ylabel(r"$\delta(x)$ approximation")
ax.grid(True)
ax.legend()

ax_L = plt.axes([0.1, 0.1, 0.8, 0.03])
slider_L = Slider(ax_L, 'L (1/ε)', 1, 10000, valinit=initial_L, valstep=1)


def update(val):
    L = slider_L.val
    epsilon = 1 / L

    y_sinc = delta_sinc(x, L)
    y_gauss = delta_gaussian(x, epsilon)

    line_sinc.set_ydata(y_sinc)
    line_gauss.set_ydata(y_gauss)

    y_max = max(np.max(y_sinc), np.max(y_gauss))
    ax.set_ylim(-0.25 * y_max, 1.2 * y_max)
    fig.canvas.draw_idle()


slider_L.on_changed(update)

plt.show()
