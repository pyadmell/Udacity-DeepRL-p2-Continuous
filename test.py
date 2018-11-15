"""
"""

import numpy as np
import torch
from PPO import GaussianActorCriticNetwork

def main():
    model = GaussianActorCriticNetwork(state_dim=4, action_dim=4)
    xvals = np.random.randn(100, 4)
    x = torch.from_numpy(xvals).float()
    acts = np.random.randn(100, 4)
    actions = torch.from_numpy(xvals).float()
    print(" ---- x")
    print(f"   shape = {x.shape}")
    print(x[:5, :])
    print(" ---- actions")
    print(actions[:5, :])
    print(f"   shape = {actions.shape}")
    actions, log_probs, v = model.forward(x, actions)
    print(" ---- actions")
    print(actions[:5, :])
    print(f"   shape = {actions.shape}")
    print(" ---- log_probs")
    print(log_probs[:5, :])
    print(f"   shape = {log_probs.shape}")
    print(" ---- v")
    print(v[:5, :])
    print(f"   shape = {v.shape}")

if __name__ == "__main__":
    main()
