# src/seobm/seobm_tensor.py
# ======================================
# SEOBM Tensor Construction
# ======================================

import numpy as np

class SEOBM:
    def __init__(self, num_features, window_size):
        self.F = num_features
        self.K = window_size
        self.tensor = None

    def build(self, feature_sequence):
        """
        Args:
            feature_sequence : (T,F)
        Returns:
            seobm : (T,F,K)
        """
        T = feature_sequence.shape[0]
        seobm = np.zeros((T, self.F, self.K))

        for t in range(T):
            for k in range(self.K):
                if t - k >= 0:
                    seobm[t, :, k] = feature_sequence[t - k]

        self.tensor = seobm
        return seobm
