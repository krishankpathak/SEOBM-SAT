# src/models/seobm_lstm.py
# ======================================
# SEOBM-based temporal predictor
# ======================================

import torch
import torch.nn as nn

class SEOBMLSTM(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=32,
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, K, F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
