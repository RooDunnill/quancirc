import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


x_vals = np.linspace(-10, 10, 500)

t_init = 0.0
m_init = 1.0

def scalar_propagator(x, t, m):
    s = np.sqrt(np.maximum(t**2 - x**2, 1e-10))  
    return np.where(t > np.abs(x), np.sin(m * s) / s, 0)


fig, ax = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(bottom=0.25)

line, = ax.plot(x_vals, scalar_propagator(x_vals, t_init, m_init), lw=2)
ax.set_title("Klein Gordon Delayed Propagator G(x, t)")
ax.set_xlabel("x")
ax.set_ylabel("G(x, t)")
ax.set_ylim(-1.2, 1.2)
ax.grid(True)


ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_mass = plt.axes([0.2, 0.05, 0.65, 0.03])

slider_time = Slider(ax_time, 'Time (t)', 0.01, 10.0, valinit=t_init, valstep=0.01)
slider_mass = Slider(ax_mass, 'Mass (m)', 0.01, 10.0, valinit=m_init, valstep=0.10)

# Update function
def update(val):
    t = slider_time.val
    m = slider_mass.val
    ax.set_ylim(-1.2 * m, 1.2 * m)
    line.set_ydata(scalar_propagator(x_vals, t, m))
    fig.canvas.draw_idle()

slider_time.on_changed(update)
slider_mass.on_changed(update)

plt.show()
