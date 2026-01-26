# src/run_mappo_seobm_demo.py
# ======================================
# MAPPO + SEOBM demonstration
# ======================================

import os, sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.envs.orbital_marl_env import OrbitalMARLEnv
from src.models.seobm_encoder import SEOBMEncoder
from src.models.mappo_actor_critic import MAPPOActor

# -------- Setup --------
N = 3
F = 4
K = 10

env = OrbitalMARLEnv(num_agents=N)
encoder = SEOBMEncoder(num_features=F)
actors = [MAPPOActor(obs_dim=64, action_dim=3) for _ in range(N)]

# -------- Rollout --------
obs = env.reset()

for step in range(10):
    # Fake SEOBM input (normally comes from pipeline)
    seobm = torch.randn(N, K, F)

    actions = []
    for i in range(N):
        z = encoder(seobm[i:i+1])
        a = actors[i](z).detach().numpy().squeeze()
        actions.append(a)

    actions = np.array(actions)
    obs, reward, done, _ = env.step(actions)

    print(f"Step {step}, reward {reward:.3f}")

print("✅ MAPPO + SEOBM demo complete.")
