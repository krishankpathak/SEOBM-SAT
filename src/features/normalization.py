# src/features/normalization.py
# ======================================
# Feature normalization utilities
# ======================================

import numpy as np

def normalize_features(features, eps=1e-8):
    """
    Z-score normalization per feature.

    Args:
        features : (T,F)

    Returns:
        norm_features, mean, std
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0) + eps
    return (features - mean) / std, mean, std
