# src/physics/velocity_reconstruction.py
# ======================================
# Layer-1 base state construction
# r(t) -> (r(t), v(t), t)
# ======================================

import os
import numpy as np

DEFAULT_DT = 60.0  # seconds (assumed uniform sampling)


def reconstruct_velocity(r: np.ndarray, dt: float = DEFAULT_DT):
    """
    Finite-difference velocity reconstruction.

    Args:
        r  : (T,3) ECI positions [km]
        dt : timestep [s]

    Returns:
        v  : (T,3) velocities [km/s]
        t  : (T,) time array [s]
    """
    T = r.shape[0]
    v = np.zeros_like(r)

    v[1:] = (r[1:] - r[:-1]) / dt
    v[0] = v[1]  # pad first step

    t = np.arange(T) * dt
    return v, t


def build_base_state(parsed_npz_path, out_path, dt=DEFAULT_DT):
    """
    Build and save base state from parsed Layer-2 file.
    """
    data = np.load(parsed_npz_path)
    r = data["r"]

    v, t = reconstruct_velocity(r, dt)

    np.savez(out_path, r=r, v=v, t=t)
    return out_path
