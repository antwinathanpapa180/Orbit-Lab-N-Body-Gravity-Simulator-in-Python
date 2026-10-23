"""
nbody.py - vectorized N-body core (2D).

State convention:
    pos : (N, 2) array     vel : (N, 2) array     masses : (N,) array
Every integrator has the signature  step(pos, vel, dt, acc_fn) -> (pos, vel)
where acc_fn(pos) -> (N, 2) accelerations.
"""
import numpy as np


# ---------- Physics (fully vectorized, no Python loops over bodies) ----------
def make_accel(masses, G=1.0, eps=0.0):
    """Return acc_fn(pos). eps is a Plummer softening length."""
    masses = np.asarray(masses, dtype=float)

    def acc_fn(pos):
        # diff[i, j] = pos[j] - pos[i]  -> shape (N, N, 2)
        diff = pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff) + eps**2
        np.fill_diagonal(r2, 1.0)            # dummy value, avoids 1/0
        inv_r3 = r2 ** -1.5
        np.fill_diagonal(inv_r3, 0.0)        # no self-force
        return G * np.einsum("ij,ijk->ik", inv_r3 * masses[None, :], diff)

    return acc_fn


def total_energy(pos, vel, masses, G=1.0, eps=0.0):
    ke = 0.5 * np.sum(masses * np.einsum("ik,ik->i", vel, vel))
    diff = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff) + eps**2)
    iu = np.triu_indices(len(masses), k=1)
    pe = -G * np.sum(masses[iu[0]] * masses[iu[1]] / r[iu])
    return ke + pe


def angular_momentum(pos, vel, masses):
    return np.sum(masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))


def total_momentum(vel, masses):
    return np.sum(masses[:, None] * vel, axis=0)


def to_com_frame(pos, vel, masses):
    """Shift to centre-of-mass frame with zero net momentum."""
    m = masses[:, None]
    pos = pos - np.sum(m * pos, axis=0) / masses.sum()
    vel = vel - np.sum(m * vel, axis=0) / masses.sum()
    return pos, vel


# ---------- Integrators ----------
def euler(pos, vel, dt, acc_fn):
    return pos + vel * dt, vel + acc_fn(pos) * dt


def symplectic_euler(pos, vel, dt, acc_fn):
    vel_new = vel + acc_fn(pos) * dt
    return pos + vel_new * dt, vel_new


def velocity_verlet(pos, vel, dt, acc_fn):
    a = acc_fn(pos)
    pos_new = pos + vel * dt + 0.5 * a * dt**2
    vel_new = vel + 0.5 * (a + acc_fn(pos_new)) * dt
    return pos_new, vel_new


def rk4(pos, vel, dt, acc_fn):
    k1p, k1v = vel, acc_fn(pos)
    k2p, k2v = vel + 0.5 * dt * k1v, acc_fn(pos + 0.5 * dt * k1p)
    k3p, k3v = vel + 0.5 * dt * k2v, acc_fn(pos + 0.5 * dt * k2p)
    k4p, k4v = vel + dt * k3v, acc_fn(pos + dt * k3p)
    pos_new = pos + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
    vel_new = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return pos_new, vel_new


INTEGRATORS = {
    "Euler": euler,
    "Symplectic Euler": symplectic_euler,
    "Velocity Verlet": velocity_verlet,
    "RK4": rk4,
}


# ---------- Driver ----------
def simulate(step_fn, pos0, vel0, masses, dt, n_steps, G=1.0, eps=0.0, record_every=1):
    """Run a simulation. Returns dict with times, positions, energy, L, momentum."""
    acc_fn = make_accel(masses, G, eps)
    pos, vel = pos0.copy(), vel0.copy()
    n_rec = n_steps // record_every + 1
    N = len(masses)
    out = {
        "t": np.empty(n_rec),
        "pos": np.empty((n_rec, N, 2)),
        "E": np.empty(n_rec),
        "L": np.empty(n_rec),
        "P": np.empty(n_rec),
    }

    def record(k, i):
        out["t"][k] = i * dt
        out["pos"][k] = pos
        out["E"][k] = total_energy(pos, vel, masses, G, eps)
        out["L"][k] = angular_momentum(pos, vel, masses)
        out["P"][k] = np.linalg.norm(total_momentum(vel, masses))

    record(0, 0)
    for i in range(1, n_steps + 1):
        pos, vel = step_fn(pos, vel, dt, acc_fn)
        if i % record_every == 0:
            record(i // record_every, i)
    return out
