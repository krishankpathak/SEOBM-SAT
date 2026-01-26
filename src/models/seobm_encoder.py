# src/models/seobm_encoder.py
# ======================================
# SEOBM Encoder Network
# ======================================

import torch
import torch.nn as nn

class SEOBMEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, seobm):
        # seobm: (B,K,F)
        _, (h, _) = self.lstm(seobm)
        return h[-1]
