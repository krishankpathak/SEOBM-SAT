# src/physics/propagation.py
# ======================================
# Physics-consistent synthetic propagation
# ======================================

import numpy as np

MU_EARTH = 398600.4418  # km^3 / s^2


def two_body_acceleration(r):
    """
    Two-body gravitational acceleration.
    """
    norm = np.linalg.norm(r)
    return -MU_EARTH * r / (norm**3)


def propagate_two_body(r0, v0, dt, steps):
    """
    Simple two-body propagation using Euler integration.

    Args:
        r0    : initial position (3,)
        v0    : initial velocity (3,)
        dt    : timestep [s]
        steps : number of steps

    Returns:
        r, v : propagated states
    """
    r = np.zeros((steps, 3))
    v = np.zeros((steps, 3))

    r[0], v[0] = r0, v0

    for k in range(steps - 1):
        a = two_body_acceleration(r[k])
        v[k + 1] = v[k] + a * dt
        r[k + 1] = r[k] + v[k + 1] * dt

    return r, v


def generate_synthetic_trajectory(base_state_npz, out_path,
                                  dt=10.0, steps=1000,
                                  noise_pos=0.01, noise_vel=1e-4):
    """
    Generate synthetic trajectory from base state.
    """
    base = np.load(base_state_npz)
    r0 = base["r"][0]
    v0 = base["v"][0]

    r, v = propagate_two_body(r0, v0, dt, steps)

    # Controlled stochastic perturbations
    r += np.random.normal(0, noise_pos, r.shape)
    v += np.random.normal(0, noise_vel, v.shape)

    np.savez(out_path, r=r, v=v)
    return out_path
