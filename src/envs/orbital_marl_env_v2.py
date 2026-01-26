# src/envs/orbital_marl_env_v2.py
# ======================================
# Learnable synthetic MARL orbital env
# ======================================

import numpy as np

class OrbitalMARLEnvV2:
    def __init__(self, num_agents=3, dt=0.5, min_dist=2.0):
        self.N = num_agents
        self.dt = dt
        self.min_dist = min_dist
        self.reset()

    def reset(self):
        self.r = np.random.uniform(-5, 5, (self.N, 3))
        self.v = np.zeros((self.N, 3))
        return self._obs()

    def step(self, actions):
        self.v += actions
        self.r += self.v * self.dt

        reward = self._reward()
        done = self._done()

        return self._obs(), reward, done, {}

    def _obs(self):
        return self.r.copy(), self.v.copy()

    def _reward(self):
        reward = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = np.linalg.norm(self.r[i] - self.r[j])
                if d < self.min_dist:
                    reward -= 10.0
                else:
                    reward += 1.0 / d
        return reward

    def _done(self):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if np.linalg.norm(self.r[i] - self.r[j]) < 0.5:
                    return True
        return False
