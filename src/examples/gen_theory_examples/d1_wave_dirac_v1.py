import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import random


hbar_GeVs = 6.58211957e-25  # hbar in GeV·s
gev_to_mev = 1000
gev_inv_to_fm = 0.197327

L = 4.0      #GeV-1 
Nx = 4000            
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx, dtype=np.float32)

phi_prev = np.zeros(Nx, dtype=np.float32)
phi_now  = np.zeros(Nx, dtype=np.float32)
phi_next = np.zeros(Nx, dtype=np.float32)


c = 1.0              
mass = 0.5          #GeV
T_max = 10.0
dt = 0.5 * dx / c       # CFL condition and GeV-1
steps = int(T_max / dt)
frame_stride = 25

C2 = (c * dt / dx) ** 2

config = {
    "phi3": True,
    "phi4": True,
    "double_well": False,
    "higgs": False,
    "lamb": 0.0001,
    "mu": 0.001,
    "vev": 100.0
}

def dV_dphi(phi, cfg):
    total = 0.0
    if cfg["phi3"]:
        total += cfg["lamb"] * phi**2
    if cfg["phi4"]:
        total += cfg["lamb"] * phi**3
    if cfg["double_well"]:
        total += cfg["mu"]**2 * phi - cfg["lamb"] * phi**3
    if cfg["higgs"]:
        total += cfg["lamb"] * phi * (phi**2 - cfg["vev"]**2)
    return total

n_packets = 20
packet_lifetime = 1
packet_sigma = 0.1
packet_amp = 50.0

packets = [{"t0": 0, "x0": 0.5 * L, "omega": 11 * np.pi}]
for _ in range(n_packets):
    t0 = np.random.randint(0, (steps - packet_lifetime) // 10)
    x0 = np.random.uniform(0.2 * L, 0.8 * L)
    freq = np.random.uniform(3.0, 8.0)
    omega = 2 * np.pi * freq
    packets.append({"t0": t0, "x0": x0, "omega": omega})




frames = []
energy_log = []

def update_phi(phi_prev, phi_now, phi_next, source, C2, dt, mass):
    phi_next[1:-1] = (2 * phi_now[1:-1] - phi_prev[1:-1] + C2 * (phi_now[2:] - 2 * phi_now[1:-1] + phi_now[:-2])
        + dt**2 * (source[1:-1] - mass**2 * phi_now[1:-1] - dV_dphi(phi_now[1:-1], config)))


for n in range(steps):
    t = n * dt
    source = np.zeros(Nx, dtype=np.float32)

    for p in packets:
        if p["t0"] <= n < p["t0"] + packet_lifetime:
            envelope = np.exp(-((x - p["x0"])**2) / (2 * packet_sigma**2))
            source += packet_amp * envelope * np.cos(p["omega"] * t)

        update_phi(phi_prev, phi_now, phi_next, source, C2, dt, mass)
    phi_next[0] = phi_next[-1] = 0  #


    if n % frame_stride == 0:

        dphi_dt = (phi_next - phi_now) / dt

        dphi_dx = np.zeros_like(phi_now)
        dphi_dx[1:-1] = (phi_now[2:] - phi_now[:-2]) / (2 * dx)

        energy_density = 0.5 * (dphi_dt**2 +(c**2) * dphi_dx**2 +(mass**2) * phi_now**2)

        total_energy = np.sum(energy_density) * dx
        energy_log.append(total_energy)

        frames.append(phi_now.copy())

    print(f"\rStep {n} out of {steps} completed", end="")
    phi_prev[:], phi_now[:] = phi_now, phi_next



fig, ax = plt.subplots(figsize=(12,4))
line, = ax.plot(x, frames[0])
label = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, va='top')

graph_lim = 1.2 * max(np.max(np.abs(frame)) for frame in frames)
ax.set_ylim(-graph_lim, graph_lim)
ax.set_xlim(0, L)
ax.set_title("1D Massive Wave Equation with Particle Creation")
ax.set_xlabel("x")
ax.set_ylabel("ϕ(x, t)")

def animate(i):
    line.set_ydata(frames[i])
    current_time = i * dt * frame_stride
    current_energy = energy_log[i]
    label.set_text(f"t = {current_time:.2f} s\nE = {current_energy:.7f}")
    return [line, label]


interval_ms = 1000 * dt * frame_stride
ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=interval_ms, blit=True)
plt.show()
