# src/physics/noise_models.py
# ======================================
# Noise & maneuver primitives
# ======================================

import numpy as np


def gaussian_position_noise(r, sigma_km=0.01):
    return r + np.random.normal(0, sigma_km, r.shape)


def impulsive_maneuver(v, dv_max=0.01, probability=0.01):
    """
    Random impulsive delta-v.
    """
    if np.random.rand() < probability:
        dv = np.random.uniform(-dv_max, dv_max, size=3)
        return v + dv
    return v
