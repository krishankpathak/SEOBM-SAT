# src/run_seobm_build.py
# ======================================
# Build SEOBM tensors
# ======================================

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.seobm.seobm_tensor import SEOBM

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer1_synthetic", "features"
)

SEOBM_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer1_synthetic", "seobm_tensors"
)
os.makedirs(SEOBM_DIR, exist_ok=True)

feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")]
assert feature_files, "No feature files found."

features = np.load(os.path.join(FEATURE_DIR, feature_files[0]))

model = SEOBM(num_features=features.shape[1], window_size=10)
seobm_tensor = model.build(features)

out_path = os.path.join(SEOBM_DIR, "seobm_test.npy")
np.save(out_path, seobm_tensor)

print("SEOBM shape:", seobm_tensor.shape)
print("Saved:", out_path)
