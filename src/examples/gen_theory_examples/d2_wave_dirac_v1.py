import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange
from numba import threading_layer
import random

Lx, Ly = 2.0, 2.0      
Nx, Ny = 240, 120    
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx, dtype=np.float32)
y = np.linspace(0, Ly, Ny, dtype=np.float32)
X, Y = np.meshgrid(x, y, indexing='ij')

c = 2.0                
T_max = 4.0         
dt = 0.4 * min(dx, dy) / c  
steps = int(T_max / dt)


mass = 2.0
n_packets = 5
packet_lifetime = 2
packet_sigma = 0.25
packet_amp = 200.0

packets = []

for _ in range(n_packets):
    t0 = np.random.randint(0, (steps - packet_lifetime)/5)
    x0 = np.random.uniform(0.2 * Lx, 0.8 * Lx)
    y0 = np.random.uniform(0.2 * Ly, 0.8 * Ly)
    freq = np.random.uniform(3.0, 8.0)
    omega = 2 * np.pi * freq
    packets.append({
        "t0": t0,
        "x0": x0,
        "y0": y0,
        "omega": omega
    })


phi_prev = np.zeros((Nx, Ny), dtype=np.float32)
phi_now = np.zeros((Nx, Ny), dtype=np.float32)
phi_next = np.zeros((Nx, Ny), dtype=np.float32)

frames = []


C2x = (c * dt / dx) ** 2
C2y = (c * dt / dy) ** 2

Nx, Ny = phi_now.shape

@njit
def update_phi(phi_prev, phi_now, phi_next, source, C2x, C2y, dt):
    phi_next[1:-1, 1:-1] = (
        2 * phi_now[1:-1, 1:-1]
        - phi_prev[1:-1, 1:-1]
        + C2x * (phi_now[2:, 1:-1] - 2 * phi_now[1:-1, 1:-1] + phi_now[:-2, 1:-1])
        + C2y * (phi_now[1:-1, 2:] - 2 * phi_now[1:-1, 1:-1] + phi_now[1:-1, :-2])
        + dt**2 * (source[1:-1, 1:-1] - mass**2 * phi_now[1:-1, 1:-1])
    )

for n in range(steps):
    t = n * dt

    source = np.zeros_like(phi_now)

    for p in packets:
        if p["t0"] <= n < p["t0"] + packet_lifetime:
            envelope = np.exp(-((X - p["x0"])**2 + (Y - p["y0"])**2) / (2 * packet_sigma**2))
            source += packet_amp * envelope * np.sin(p["omega"] * t)
        
    update_phi(phi_prev, phi_now, phi_next, source, C2x, C2y, dt)

    phi_next[0, :] = phi_next[-1, :] = 0
    phi_next[:, 0] = phi_next[:, -1] = 0


    phi_prev[:, :], phi_now[:, :] = phi_now, phi_next

    if n % 5 == 0:
        frames.append(phi_now.copy())


fig, ax = plt.subplots()
im = ax.imshow(frames[0], extent=(0, Lx, 0, Ly), origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1)
ax.set_title("2D Wave Equation with Spontaneous Creation of Particles")
ax.set_xlabel("x")
ax.set_ylabel("y")

def animate(i):
    im.set_array(frames[i])
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=24, blit=True)
plt.show()
