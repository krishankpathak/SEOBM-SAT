# src/ablations/memory_ablation.py
# ======================================
# Memory window ablation for SEOBM
# ======================================

import numpy as np

def ablate_memory(seobm_tensor, k_keep):
    """
    Remove long-term memory beyond k_keep.

    Args:
        seobm_tensor : (T,F,K)
        k_keep       : int

    Returns:
        ablated tensor
    """
    ablated = seobm_tensor.copy()
    ablated[:, :, k_keep:] = 0.0
    return ablated
