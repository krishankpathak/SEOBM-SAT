# src/run_feature_extraction.py
# ======================================
# Generate behavior feature dataset
# ======================================

import os
import sys
import numpy as np

# ---- Fix path ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.features.behavior_features import extract_behavior_features

TRAJ_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer1_synthetic", "trajectories"
)

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer1_synthetic", "features"
)
os.makedirs(FEATURE_DIR, exist_ok=True)

traj_files = [f for f in os.listdir(TRAJ_DIR) if f.endswith(".npz")]

assert traj_files, "No trajectories found."

for file in traj_files:
    data = np.load(os.path.join(TRAJ_DIR, file))
    r, v = data["r"], data["v"]

    features = extract_behavior_features(r, v)

    out_path = os.path.join(
        FEATURE_DIR, file.replace("traj", "features")
    )
    np.save(out_path, features)

    print(f"Saved features: {out_path} | shape {features.shape}")

print("✅ Feature extraction complete.")
