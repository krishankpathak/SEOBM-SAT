# src/run_mappo_seobm_train.py
# ======================================
# Trained MARL with SEOBM (demo)
# ======================================

import os, sys
import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.envs.orbital_marl_env_v2 import OrbitalMARLEnvV2
from src.models.seobm_encoder import SEOBMEncoder
from src.models.mappo_actor_critic import MAPPOActor
from src.rl.simple_policy_gradient import policy_gradient_step

# -------- Setup --------
N, F, K = 3, 4, 10
env = OrbitalMARLEnvV2(num_agents=N)
encoder = SEOBMEncoder(num_features=F)

actors = [MAPPOActor(obs_dim=64, action_dim=3) for _ in range(N)]
opts = [optim.Adam(a.parameters(), lr=1e-3) for a in actors]

EPISODES = 50
MAX_STEPS = 25

# -------- Training --------
for ep in range(EPISODES):
    env.reset()
    episode_rewards = []
    log_probs = [[] for _ in range(N)]

    for step in range(MAX_STEPS):
        seobm = torch.randn(N, K, F)  # placeholder for real SEOBM
        actions = []

        for i in range(N):
            z = encoder(seobm[i:i+1])
            mu = actors[i](z)
            dist = torch.distributions.Normal(mu, 0.5)
            a = dist.sample()
            log_probs[i].append(dist.log_prob(a).sum())
            actions.append(a.detach().numpy().squeeze())

        _, reward, done, _ = env.step(np.array(actions))
        episode_rewards.append(reward)

        if done:
            break

    for i in range(N):
        policy_gradient_step(
            actors[i], opts[i], log_probs[i], episode_rewards
        )

    if ep % 10 == 0:
        print(f"Episode {ep}, total reward {sum(episode_rewards):.2f}")

print("✅ Trained MAPPO + SEOBM experiment complete.")
