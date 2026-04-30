"""Reference utilities for the viscous Burgers equation.

Problem used throughout the thesis project:

    u_t + u u_x - nu u_xx = 0,   x in [-1, 1], t in [0, 1]
    u(x, 0) = -sin(pi x)
    u(-1, t) = u(1, t) = 0

The common PINN benchmark does not use a simple elementary closed form for
this initial-boundary-value problem.  We therefore generate a high-resolution
finite-difference reference and compare all learned/surrogate methods against
that same reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BurgersDomain:
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    nu: float = 0.01 / np.pi


def initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def boundary_condition(t: np.ndarray) -> np.ndarray:
    return np.zeros_like(t)


def solve_burgers_fd(
    x: np.ndarray,
    t_eval: np.ndarray,
    nu: float = 0.01 / np.pi,
    cfl: float = 0.35,
    diffusion_safety: float = 0.20,
) -> np.ndarray:
    """Solve Burgers on a fixed uniform grid using an explicit stable scheme.

    The nonlinear convection term is discretized with first-order upwinding and
    the diffusion term with a centered second derivative.  The routine evolves
    once and stores the solution at all requested times, so it is much faster
    than restarting from t=0 for every time slice.
    """

    x = np.asarray(x, dtype=np.float64).ravel()
    t_eval = np.asarray(t_eval, dtype=np.float64).ravel()
    if x.ndim != 1 or t_eval.ndim != 1:
        raise ValueError("x and t_eval must be one-dimensional arrays")
    if len(x) < 3:
        raise ValueError("x grid must contain at least three points")
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-8, atol=1e-10):
        raise ValueError("solve_burgers_fd expects a uniform x grid")

    order = np.argsort(t_eval)
    t_sorted = t_eval[order]
    u = initial_condition(x).astype(np.float64)
    u[0] = 0.0
    u[-1] = 0.0

    snapshots = np.empty((x.size, t_eval.size), dtype=np.float64)
    current_t = 0.0
    next_slot = 0

    while next_slot < t_sorted.size and np.isclose(t_sorted[next_slot], 0.0):
        snapshots[:, order[next_slot]] = u
        next_slot += 1

    while next_slot < t_sorted.size:
        target_t = float(t_sorted[next_slot])
        while current_t < target_t - 1e-14:
            umax = max(1e-8, float(np.max(np.abs(u))))
            dt_conv = cfl * dx / umax
            dt_diff = diffusion_safety * dx * dx / max(float(nu), 1e-12)
            dt = min(dt_conv, dt_diff, target_t - current_t)

            old = u.copy()
            old[0] = 0.0
            old[-1] = 0.0

            dudx = np.zeros_like(old)
            center = old[1:-1]
            left = old[:-2]
            right = old[2:]
            dudx[1:-1] = np.where(center >= 0.0, (center - left) / dx, (right - center) / dx)
            d2udx2 = (right - 2.0 * center + left) / (dx * dx)

            u[1:-1] = center - dt * center * dudx[1:-1] + nu * dt * d2udx2
            u[0] = 0.0
            u[-1] = 0.0
            current_t += dt

        snapshots[:, order[next_slot]] = u
        next_slot += 1

    return snapshots


def grid_metrics(u_true: np.ndarray, u_pred: np.ndarray) -> dict[str, float]:
    diff = np.asarray(u_pred) - np.asarray(u_true)
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    linf = float(np.max(np.abs(diff)))
    rel_l2 = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(np.asarray(u_true).ravel()), 1e-12))
    return {"mse": mse, "mae": mae, "linf": linf, "rel_l2": rel_l2}


def make_reference_grid(nx: int = 128, nt: int = 81, domain: BurgersDomain = BurgersDomain()):
    x = np.linspace(domain.x_min, domain.x_max, nx)
    t = np.linspace(domain.t_min, domain.t_max, nt)
    u = solve_burgers_fd(x, t, nu=domain.nu)
    return x, t, u


def save_reference(path: str | Path, nx: int = 128, nt: int = 81) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    domain = BurgersDomain()
    x, t, u = make_reference_grid(nx=nx, nt=nt, domain=domain)
    np.savez(path, x=x, t=t, u=u, nu=domain.nu)
    return path
