"""
Examples: figure-8 three-body orbit and the Solar System.
Run:  python run_examples.py          (saves images in images/, then shows plots)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from nbody import INTEGRATORS, simulate, to_com_frame

# Results are saved in the 'images' folder next to this script
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"Euler": "tab:red", "Symplectic Euler": "tab:orange",
          "Velocity Verlet": "tab:green", "RK4": "tab:blue"}


# =====================================================================
# 1. Figure-8 three-body orbit (Chenciner & Montgomery, 2000), G = m = 1
# =====================================================================
def figure8_ic():
    p1 = np.array([-0.97000436, 0.24308753])
    v3 = np.array([-0.93240737, -0.86473146])
    pos = np.array([p1, -p1, [0.0, 0.0]])
    vel = np.array([-v3 / 2, -v3 / 2, v3])
    return pos, vel, np.ones(3)


def run_figure8():
    pos0, vel0, m = figure8_ic()
    T = 6.32591398                    # known period of the orbit
    dt = 0.002
    n_periods = 10
    n_steps = int(n_periods * T / dt)

    results = {}
    for name, fn in INTEGRATORS.items():
        print(f"  figure-8: {name}")
        results[name] = simulate(fn, pos0, vel0, m, dt, n_steps, record_every=10)

    # Energy error
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        err = np.abs((r["E"] - r["E"][0]) / r["E"][0]) + 1e-18
        ax.semilogy(r["t"] / T, err, label=name, color=COLORS[name])
    ax.set_xlabel("time (periods)")
    ax.set_ylabel("|ΔE / E0|")
    ax.set_title(f"Figure-8 three-body: energy error (dt = {dt})")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "figure8_energy.png"), dpi=150)

    # Trajectories
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        for b in range(3):
            ax.plot(r["pos"][:, b, 0], r["pos"][:, b, 1], lw=0.7)
        ax.set_title(name); ax.set_aspect("equal")
    fig.suptitle(f"Figure-8 after {n_periods} periods")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "figure8_orbits.png"), dpi=150)

    # Sensitivity demo: nudge one body by 1e-3 and watch the stars drift out of step
    pos_p = pos0.copy(); pos_p[2, 0] += 1e-3
    perturbed = simulate(INTEGRATORS["RK4"], pos_p, vel0, m, dt, int(15 * T / dt), record_every=10)
    clean = simulate(INTEGRATORS["RK4"], pos0, vel0, m, dt, int(15 * T / dt), record_every=10)
    return results, perturbed, clean, T


def animate_figure8(perturbed, clean, T):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    data = [("Exact initial conditions", clean), ("Body 3 nudged by 0.001", perturbed)]
    stride = 4
    n_frames = len(clean["t"]) // stride
    cols = ["tab:blue", "tab:orange", "tab:green"]
    dots, trails = [], []
    for ax, (title, r) in zip(axes, data):
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
        ax.set_title(title)
        trails.append([ax.plot([], [], color=c, lw=0.8, alpha=0.6)[0] for c in cols])
        dots.append([ax.plot([], [], "o", color=c, ms=8)[0] for c in cols])
    label = fig.text(0.5, 0.02, "", ha="center")

    def update(f):
        k = f * stride
        for (title, r), tr, dt_ in zip(data, trails, dots):
            lo = max(0, k - 150)
            for b in range(3):
                tr[b].set_data(r["pos"][lo:k + 1, b, 0], r["pos"][lo:k + 1, b, 1])
                dt_[b].set_data([r["pos"][k, b, 0]], [r["pos"][k, b, 1]])
        label.set_text(f"t = {clean['t'][k] / T:.1f} periods")
        return [a for g in trails + dots for a in g]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=30, blit=False)
    anim.save(os.path.join(OUT_DIR, "figure8_nudge.gif"), writer=PillowWriter(fps=30))
    plt.close(fig)


# =====================================================================
# 2. Solar System (units: AU, years, solar masses -> G = 4*pi^2)
# =====================================================================
def solar_system_ic():
    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    a = np.array([0, 0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07])
    m = np.array([1.0, 1.66e-7, 2.45e-6, 3.0e-6, 3.23e-7, 9.54e-4, 2.86e-4, 4.37e-5, 5.15e-5])
    G = 4 * np.pi**2
    rng = np.random.default_rng(42)
    ang = rng.uniform(0, 2 * np.pi, len(a))          # random starting angles
    pos = np.stack([a * np.cos(ang), a * np.sin(ang)], axis=1)
    # circular-orbit speed around the Sun: v = sqrt(G M / r)
    speed = np.where(a > 0, np.sqrt(G * m[0] / np.where(a > 0, a, 1)), 0.0)
    vel = np.stack([-speed * np.sin(ang), speed * np.cos(ang)], axis=1)
    pos, vel = to_com_frame(pos, vel, m)
    return names, pos, vel, m, G


def run_solar_system():
    names, pos0, vel0, m, G = solar_system_ic()
    dt = 0.002                                       # years (~0.7 days)
    years = 50
    n_steps = int(years / dt)
    runs = {}
    for name in ["Velocity Verlet", "RK4"]:
        print(f"  solar system: {name}")
        runs[name] = simulate(INTEGRATORS[name], pos0, vel0, m, dt, n_steps, G=G, record_every=10)

    # Orbit plot (inner planets zoomed + outer planets)
    r = runs["Velocity Verlet"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, lim, idx in [(axes[0], 1.8, range(1, 5)), (axes[1], 35, range(1, 9))]:
        for b in idx:
            ax.plot(r["pos"][:, b, 0], r["pos"][:, b, 1], lw=0.8, label=names[b])
        ax.plot(r["pos"][:, 0, 0], r["pos"][:, 0, 1], "o", color="gold", ms=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    axes[0].set_title("Inner planets"); axes[1].set_title("Full system")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Solar System, {years} years (Velocity Verlet)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "solar_system.png"), dpi=150)

    # Energy error, Verlet vs RK4
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in runs.items():
        err = np.abs((r["E"] - r["E"][0]) / r["E"][0]) + 1e-18
        ax.semilogy(r["t"], err, label=name, color=COLORS[name])
    ax.set_xlabel("time (years)"); ax.set_ylabel("|ΔE / E0|")
    ax.set_title("Solar System: energy error")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "solar_energy.png"), dpi=150)
    return runs


if __name__ == "__main__":
    print("Figure-8...")
    res, pert, clean, T = run_figure8()
    print("\nFigure-8 final dE/E0 after 10 periods:")
    for n, r in res.items():
        print(f"  {n:18s} {abs((r['E'][-1]-r['E'][0])/r['E'][0]):.2e}")
    print("Animating figure-8 sensitivity GIF...")
    animate_figure8(pert, clean, T)
    print("\nSolar System...")
    runs = run_solar_system()
    print("\nSolar System final dE/E0 after 50 years:")
    for n, r in runs.items():
        print(f"  {n:18s} {abs((r['E'][-1]-r['E'][0])/r['E'][0]):.2e}")
    plt.show()
