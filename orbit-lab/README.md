# Orbit Lab: N-Body Gravity Simulator in Python

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243)
![Matplotlib](https://img.shields.io/badge/plots-Matplotlib-11557c)

**Orbit Lab** is a Python gravity simulator that compares four numerical integration methods (**Euler, Symplectic Euler, Velocity Verlet and RK4**) on orbital mechanics problems, and measures which ones can be trusted using **energy conservation** and **convergence analysis**.

It simulates a two-body orbit, the famous **figure-8 three-body orbit**, and the **Solar System** (Sun plus 8 planets), using vectorized NumPy and Matplotlib.

![Figure-8 three-body orbit simulation: exact initial conditions versus one star nudged by 0.001](images/figure8_nudge.gif)

*Left: three equal stars chase each other along a figure-8 path. Right: the same setup with one star nudged by 0.001. The shape survives, but the stars slowly drift out of step.*

## Why this project?

A simulation can look right for a short time and still be wrong over a long one. Orbit Lab makes that visible. Each method is tested against a rule that real physics never breaks (total energy stays constant), so you can see exactly which algorithms stay honest.

It combines two ideas: **computer simulation** (modelling a physical system step by step through time) and **numerical methods** (measuring how accurate each step really is).

## Results at a glance

Final relative energy error, where **smaller is better** (dt = 0.002):

| Method | Order measured | Two-body orbit (20 orbits) | Figure-8 (10 periods) |
|---|---|---|---|
| Euler | 0.8 | 4.3e-1 | 3.0e-1 |
| Symplectic Euler | 2.0 (see note) | 1.1e-7 | 3.5e-5 |
| Velocity Verlet | 2.0 | 7e-15 | 4.6e-9 |
| RK4 | 4.1 | 2.2e-11 | 8.4e-12 |

*Note: on a circular orbit, Symplectic Euler's leading error cancels after exactly one lap, so it measures as second order there even though it is a first-order method. Euler's measured 0.8 is below its theoretical 1 because the smallest time steps tested were still fairly large for it. Exact numbers vary slightly by machine.*

<p align="center">
  <img src="images/energy_error.png" alt="Relative energy error over time for Euler, Symplectic Euler, Velocity Verlet and RK4" width="48%">
  <img src="images/convergence.png" alt="Convergence order of each integrator on a log-log plot" width="48%">
</p>

## Features

- Vectorized N-body gravity (NumPy broadcasting, no Python loops over bodies)
- Four interchangeable integrators sharing one simple interface
- Diagnostics: total energy, angular momentum and momentum at every step
- Convergence study that checks each method against its theoretical order
- Three experiments: two-body orbit, figure-8 three-body orbit, Solar System
- Optional softening parameter to avoid blow-ups when bodies nearly collide

## Quick start

Requires **Python 3.9 or newer** (developed and tested on Python 3.12 and 3.13).

```bash
git clone https://github.com/<your-username>/orbit-lab.git
cd orbit-lab
pip install -r requirements.txt

python two_body.py        # one planet around one star + convergence study
python run_examples.py    # figure-8 orbit + Solar System
```

All results are saved in the `images/` folder, and plot windows open at the end (close them to finish). `run_examples.py` may take a minute or two.

On a computer without a screen, skip the windows with `MPLBACKEND=Agg python run_examples.py`.

## What you get

| File | What it shows |
|---|---|
| `images/orbits.png` | The same eccentric orbit drawn by each method |
| `images/energy_error.png` | Energy error over time for each method |
| `images/convergence.png` | Error vs. time step (order of accuracy) |
| `images/figure8_orbits.png` | Figure-8 orbit after 10 periods, per method |
| `images/figure8_energy.png` | Energy error on the figure-8 problem |
| `images/figure8_nudge.gif` | Exact vs. slightly nudged figure-8 animation |
| `images/solar_system.png` | Sun and eight planets over 50 years |
| `images/solar_energy.png` | Velocity Verlet vs. RK4 on the Solar System |

## Project structure

```
orbit-lab/
├── nbody.py            # library: gravity, integrators, diagnostics
├── two_body.py         # experiment 1: two-body orbit + convergence study
├── run_examples.py     # experiments 2 and 3: figure-8 and Solar System
├── requirements.txt
├── images/             # generated results
└── docs/
    └── Orbit_Lab_Project_Manual.pdf
```

`nbody.py` is a library used by `run_examples.py`, so keep it in the same folder. You do not run it directly.

## How it works

Newton's law of gravitation gives the pull between two bodies:

```
F = G * m1 * m2 / r^2
```

At each step the simulator computes the pull on every body, then moves all bodies forward by a small time step `dt`. The rule used to move forward is the **integrator**:

| Integrator | Idea | Behaviour over long runs |
|---|---|---|
| Euler | Move with current speed, then update speed | Gains energy, orbits spiral outward |
| Symplectic Euler | Update speed first, then move | Bounded energy error |
| Velocity Verlet | Average the pull at start and end of the step | Bounded and very small error |
| RK4 | Blend four samples of the pull inside the step | Very accurate, error slowly accumulates |

Energy is the report card: in an isolated system it should never change, so any drift measures numerical error.

## Key findings

1. **Plain Euler is unsuitable for orbits.** It steadily gains energy and the orbit drifts away from the truth.
2. **RK4 wins the sprint, Velocity Verlet wins the marathon.** Over 50 Solar System years, RK4's energy error (2.8e-8) ended larger than Verlet's (3.4e-9), because RK4's error keeps growing while Verlet's stays bounded. This is why symplectic methods are preferred for long-term orbit studies.
3. **The figure-8 is delicate, not fragile.** Nudging one star by 0.001 leaves the shape intact but the stars slowly drift out of step (the gap between the exact and nudged runs grew from about 0.08 after 4 laps to about 0.8 after 40).

## Limitations

- 2D only, with point masses and Newtonian gravity (no relativity, no collisions)
- The Solar System uses circular orbits at random starting angles, not real-date positions, so it is not for real-world navigation
- Each step checks every pair of bodies, so cost grows with the square of the number of bodies (good for tens of bodies, not thousands)

## Roadmap

- [ ] Automated tests (Kepler period check, energy-bound check)
- [ ] 3D version with 3D animation
- [ ] Adaptive time step (for example Dormand-Prince RK45)
- [ ] Benchmark against `scipy.integrate.solve_ivp`
- [ ] Barnes-Hut algorithm and a galaxy-collision demo
- [ ] Long Solar System run (5,000+ years) to show RK4's drift overtaking Verlet's

## Documentation

A 3-page illustrated overview for beginners is in [`docs/Orbit_Lab_Project_Manual.pdf`](docs/Orbit_Lab_Project_Manual.pdf).

## References

- Chenciner, A. and Montgomery, R. (2000). A remarkable periodic solution of the three-body problem in the case of equal masses. *Annals of Mathematics*.
- Barnes, J. and Hut, P. (1986). A hierarchical O(N log N) force-calculation algorithm. *Nature*.
- Wisdom, J. and Holman, M. (1991). Symplectic maps for the N-body problem. *The Astronomical Journal*.


## Author

Nathan Papa

https://github.com/antwinathanpapa180/Orbit-Lab-N-Body-Gravity-Simulator-in-Python.git

## License

Released under the [MIT License](LICENSE).
