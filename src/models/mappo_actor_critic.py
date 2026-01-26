# src/models/mappo_actor_critic.py
# ======================================
# Minimal MAPPO Actor-Critic
# ======================================

import torch
import torch.nn as nn

class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)


class MAPPOCritic(nn.Module):
    def __init__(self, global_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_obs):
        return self.net(global_obs)
