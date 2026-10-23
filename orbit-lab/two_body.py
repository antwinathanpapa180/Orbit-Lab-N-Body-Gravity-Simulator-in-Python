"""
Two-body integrator comparison (normalized units: G = M = 1).

A test particle orbits a fixed central mass. We compare four integrators:
  - Explicit Euler        (order 1, energy spirals OUT)
  - Symplectic Euler      (order 1, energy error bounded)
  - Velocity Verlet       (order 2, symplectic)
  - RK4                   (order 4, tiny but secular drift)

Outputs (saved in the images/ folder):
  orbits.png, energy_error.png, convergence.png
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# Results are saved in the 'images' folder next to this script
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)

G = 1.0
M = 1.0


# ---------- Physics ----------
def accel(pos):
    r = np.linalg.norm(pos)
    return -G * M * pos / r**3


def energy(pos, vel):
    return 0.5 * np.dot(vel, vel) - G * M / np.linalg.norm(pos)


def angular_momentum(pos, vel):
    return pos[0] * vel[1] - pos[1] * vel[0]


# ---------- Integrators: each is step(pos, vel, dt) -> (pos, vel) ----------
def euler(pos, vel, dt):
    a = accel(pos)
    return pos + vel * dt, vel + a * dt


def symplectic_euler(pos, vel, dt):
    vel_new = vel + accel(pos) * dt
    return pos + vel_new * dt, vel_new


def velocity_verlet(pos, vel, dt):
    a = accel(pos)
    pos_new = pos + vel * dt + 0.5 * a * dt**2
    vel_new = vel + 0.5 * (a + accel(pos_new)) * dt
    return pos_new, vel_new


def rk4(pos, vel, dt):
    # State derivative: d(pos)/dt = vel, d(vel)/dt = accel(pos)
    k1p, k1v = vel, accel(pos)
    k2p, k2v = vel + 0.5 * dt * k1v, accel(pos + 0.5 * dt * k1p)
    k3p, k3v = vel + 0.5 * dt * k2v, accel(pos + 0.5 * dt * k2p)
    k4p, k4v = vel + dt * k3v, accel(pos + dt * k3p)
    pos_new = pos + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
    vel_new = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return pos_new, vel_new


INTEGRATORS = {
    "Euler": euler,
    "Symplectic Euler": symplectic_euler,
    "Velocity Verlet": velocity_verlet,
    "RK4": rk4,
}
COLORS = {"Euler": "tab:red", "Symplectic Euler": "tab:orange",
          "Velocity Verlet": "tab:green", "RK4": "tab:blue"}


# ---------- Simulation driver ----------
def simulate(step_fn, pos0, vel0, dt, n_steps):
    pos, vel = pos0.copy(), vel0.copy()
    traj = np.empty((n_steps + 1, 2))
    E = np.empty(n_steps + 1)
    L = np.empty(n_steps + 1)
    traj[0], E[0], L[0] = pos, energy(pos, vel), angular_momentum(pos, vel)
    for i in range(1, n_steps + 1):
        pos, vel = step_fn(pos, vel, dt)
        traj[i], E[i], L[i] = pos, energy(pos, vel), angular_momentum(pos, vel)
    return traj, E, L


def main():
    # ===== Part 1: energy drift on an eccentric orbit =====
    pos0 = np.array([1.0, 0.0])
    vel0 = np.array([0.0, 0.8])          # < 1.0 -> eccentric ellipse
    dt = 0.002
    n_orbits = 20
    # Period from Kepler's 3rd law: a = 1/(2 - v^2), T = 2*pi*a^1.5
    a_semi = 1.0 / (2.0 - np.dot(vel0, vel0))
    period = 2 * np.pi * a_semi**1.5
    n_steps = int(n_orbits * period / dt)
    t = np.arange(n_steps + 1) * dt

    results = {name: simulate(fn, pos0, vel0, dt, n_steps)
               for name, fn in INTEGRATORS.items()}

    # Orbits
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for ax, (name, (traj, _, _)) in zip(axes, results.items()):
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS[name], lw=0.6)
        ax.plot(0, 0, "o", color="gold", ms=10)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 1.5)
        ax.set_ylim(-2, 2)
    fig.suptitle(f"Same orbit, same dt = {dt}, {n_orbits} orbits")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "orbits.png"), dpi=150)

    # Energy error
    fig, ax = plt.subplots(figsize=(8, 5))
    E0 = results["Euler"][1][0]
    for name, (_, E, _) in results.items():
        err = np.abs((E - E0) / E0) + 1e-18   # avoid log(0)
        ax.semilogy(t, err, label=name, color=COLORS[name])
    ax.set_xlabel("time")
    ax.set_ylabel("|ΔE / E0|")
    ax.set_title("Relative energy error")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "energy_error.png"), dpi=150)

    # ===== Part 2: convergence study (circular orbit, exact solution known) =====
    # Circular orbit: pos(t) = (cos t, sin t). Compare after one full period.
    pos_c = np.array([1.0, 0.0])
    vel_c = np.array([0.0, 1.0])
    T = 2 * np.pi
    step_counts = np.array([50, 100, 200, 400, 800, 1600, 3200])
    dts = T / step_counts

    fig, ax = plt.subplots(figsize=(7, 5))
    print("\nConvergence slopes (log-log, expected: ~1, ~1, ~2, ~4):")
    for name, fn in INTEGRATORS.items():
        errs = []
        for n, h in zip(step_counts, dts):
            traj, _, _ = simulate(fn, pos_c, vel_c, h, int(n))
            errs.append(np.linalg.norm(traj[-1] - pos_c))   # exact = start point
        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        print(f"  {name:18s} slope = {slope:5.2f}")
        ax.loglog(dts, errs, "o-", label=f"{name} (slope {slope:.2f})", color=COLORS[name])
    ax.set_xlabel("dt")
    ax.set_ylabel("position error after one orbit")
    ax.set_title("Convergence order")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=150)

    # Quick numeric summary
    print("\nFinal relative energy error after "
          f"{n_orbits} orbits (dt = {dt}):")
    for name, (_, E, L) in results.items():
        print(f"  {name:18s} dE/E0 = {abs((E[-1]-E0)/E0):.2e}   "
              f"dL/L0 = {abs((L[-1]-L[0])/L[0]):.2e}")

    plt.show()


if __name__ == "__main__":
    main()
