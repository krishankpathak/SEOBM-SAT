# src/models/baseline_mlp.py
# ======================================
# Memoryless baseline predictor
# ======================================

import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, F)
        return self.net(x)
