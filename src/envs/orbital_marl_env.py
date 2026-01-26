# src/envs/orbital_marl_env.py
# ======================================
# Synthetic Orbital MARL Environment
# ======================================

import numpy as np

class OrbitalMARLEnv:
    def __init__(self, num_agents=3, dt=1.0):
        self.N = num_agents
        self.dt = dt
        self.reset()

    def reset(self):
        self.r = np.random.randn(self.N, 3) * 10.0
        self.v = np.random.randn(self.N, 3)
        return self._get_obs()

    def step(self, actions):
        """
        actions: (N,3) delta-v (synthetic)
        """
        self.v += actions
        self.r += self.v * self.dt

        obs = self._get_obs()
        reward = self._compute_reward()
        done = False

        return obs, reward, done, {}

    def _get_obs(self):
        return self.r.copy(), self.v.copy()

    def _compute_reward(self):
        # Soft collision penalty
        penalty = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                d = np.linalg.norm(self.r[i] - self.r[j])
                penalty -= np.exp(-d)
        return penalty
