# src/run_learning_demo.py
# ======================================
# Learning demo: baseline vs SEOBM
# ======================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.baseline_mlp import BaselineMLP
from src.models.seobm_lstm import SEOBMLSTM

# ---------------- Load data ----------------
FEATURE_DIR = os.path.join(
    PROJECT_ROOT, "data", "layer1_synthetic", "features"
)
SEOBM_DIR = os.path.join(
    PROJECT_ROOT, "data", "layer1_synthetic", "seobm_tensors"
)

features = np.load(os.path.join(FEATURE_DIR, os.listdir(FEATURE_DIR)[0]))
seobm = np.load(os.path.join(SEOBM_DIR, "seobm_test.npy"))

# Target: next-step speed
y = features[1:, 0]
x_feat = features[:-1]
x_seobm = seobm[:-1].transpose(0, 2, 1)  # (T,K,F)

# Convert to tensors
x_feat = torch.tensor(x_feat, dtype=torch.float32)
x_seobm = torch.tensor(x_seobm, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# ---------------- Models ----------------
baseline = BaselineMLP(x_feat.shape[1])
seobm_model = SEOBMLSTM(x_feat.shape[1])

criterion = nn.MSELoss()
opt_base = optim.Adam(baseline.parameters(), lr=1e-3)
opt_seobm = optim.Adam(seobm_model.parameters(), lr=1e-3)

# ---------------- Training ----------------
EPOCHS = 50

for epoch in range(EPOCHS):
    # Baseline
    opt_base.zero_grad()
    loss_base = criterion(baseline(x_feat), y)
    loss_base.backward()
    opt_base.step()

    # SEOBM
    opt_seobm.zero_grad()
    loss_seobm = criterion(seobm_model(x_seobm), y)
    loss_seobm.backward()
    opt_seobm.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:02d} | "
            f"Baseline MSE: {loss_base.item():.4f} | "
            f"SEOBM MSE: {loss_seobm.item():.4f}"
        )

print("\n✅ Learning demo complete.")
