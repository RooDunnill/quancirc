import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


Nx = 10000
L = 10
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

dt = 0.01  


def initial_wavepacket(x0=0.0, k0=5.0, sigma=0.5):
    norm = (1/(sigma * np.sqrt(np.pi)))**0.5
    return norm * np.exp(- (x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)


def V_square_well(depth=50, width=2, position=0):
    V = np.zeros_like(x)
    V[np.abs(x - position) > width / 2] = - depth
    return V

def V_square_wall(height=50, width=2, position=0):
    V = np.zeros_like(x)
    V[np.abs(x - position) > width / 2] = height
    return V

def V_harmonic(k=1.0):
    return 0.5 * k * x**2

def V_custom(k=1.0):
    return k * np.abs(x)


V = V_square_wall(height=1, width=2, position=5) + V_square_well(depth=1, width=2, position=-5)


expV = np.exp(-1j * V * dt / 2)
expK = np.exp(-1j * (k**2 / 2) * dt)


psi = initial_wavepacket()


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
wavefunc, = ax.plot(x, np.abs(psi)**2)
potential = ax.plot(x, V)
ax.set_ylim(0, 1.2 * np.max(np.abs(psi)**2))
ax.set_xlim(-10,10)
ax.set_title("Time Evolution of $|\psi(x,t)|^2$")
ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x, t)|^2$")
ax.grid(True)


ax_k0 = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_k0 = Slider(ax_k0, 'k₀ (momentum)', 0.1, 100.0, valinit=1.0)

def update(val):
    global psi
    k0 = slider_k0.val
    psi = initial_wavepacket(k0=k0)
    wavefunc.set_ydata(np.abs(psi)**2)
    fig.canvas.draw_idle()

slider_k0.on_changed(update)


def evolve(frame):
    global psi
    psi = expV * psi
    psi = np.fft.ifft(expK * np.fft.fft(psi))
    psi = expV * psi
    wavefunc.set_ydata(np.abs(psi)**2)
    return wavefunc,

ani = FuncAnimation(fig, evolve, frames=200, interval=30, blit=False)
plt.show()
