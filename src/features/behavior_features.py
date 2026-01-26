# src/features/behavior_features.py
# ======================================
# Behavior Feature Extraction
# ======================================

import numpy as np

def speed(v):
    return np.linalg.norm(v, axis=1)


def angular_momentum(r, v):
    return np.linalg.norm(np.cross(r, v), axis=1)


def radial_distance(r):
    return np.linalg.norm(r, axis=1)


def radial_velocity(r, v):
    r_hat = r / np.linalg.norm(r, axis=1, keepdims=True)
    return np.sum(v * r_hat, axis=1)


def extract_behavior_features(r, v):
    """
    Extract behavior features from trajectory.

    Args:
        r : (T,3) positions
        v : (T,3) velocities

    Returns:
        features : (T,F)
    """
    f_speed = speed(v)
    f_angmom = angular_momentum(r, v)
    f_radius = radial_distance(r)
    f_radial_v = radial_velocity(r, v)

    features = np.stack([
        f_speed,
        f_angmom,
        f_radius,
        f_radial_v
    ], axis=1)

    return features
