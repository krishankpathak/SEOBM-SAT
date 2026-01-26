# src/run_layer1_pipeline.py
# ======================================
# Test runner for Layer-1 pipeline
# ======================================

# src/run_layer1_pipeline.py
# ======================================
# Test runner for Layer-1 pipeline
# ======================================

import os
import sys

# ------------------------------------------------
# Fix Python path so `src` is importable
# ------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from src.physics.velocity_reconstruction import build_base_state
from src.physics.propagation import generate_synthetic_trajectory


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PARSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer2_public", "starlink_kaggle", "parsed_states"
)

BASE_STATE_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer1_synthetic", "base_states"
)

TRAJ_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "layer1_synthetic", "trajectories"
)

os.makedirs(BASE_STATE_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)

# --- Pick ONE parsed file ---
parsed_files = [f for f in os.listdir(PARSED_DIR) if f.endswith(".npz")]
assert parsed_files, "No parsed Layer-2 files found."

parsed_path = os.path.join(PARSED_DIR, parsed_files[0])
base_out = os.path.join(BASE_STATE_DIR, "base_test.npz")
traj_out = os.path.join(TRAJ_DIR, "traj_test.npz")

print("Building base state...")
build_base_state(parsed_path, base_out)

print("Generating synthetic trajectory...")
generate_synthetic_trajectory(base_out, traj_out)

print("✅ Layer-1 pipeline executed successfully.")
print("Base state:", base_out)
print("Trajectory:", traj_out)
