# src/tasks/delayed_prediction.py
# ======================================
# Temporal dependency task utilities
# ======================================

import numpy as np

def delayed_target(features, delay):
    """
    Create delayed prediction targets.

    Args:
        features : (T,F)
        delay    : int

    Returns:
        x, y
    """
    x = features[:-delay]
    y = features[delay:, 0]  # delayed speed
    return x, y
