# src/rl/simple_policy_gradient.py
# ======================================
# Minimal policy gradient update
# ======================================

import torch

def policy_gradient_step(actor, optimizer, log_probs, rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = 0.0
    for log_p, G in zip(log_probs, returns):
        loss -= log_p * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
