"""Gaussian ActorCritic Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FCNetwork, to_tensor


class GaussianActorCriticNetwork(nn.Module):
    """連続分布を出力するような Actor-Critic Network モデル
    Actor としては State から連続を出すようなモデル
    Critic としては State から Value を出力するモデル
    """
    def __init__(self, state_dim=1, action_dim=1,
        hiddens_actor=[64, 64], hiddens_critic=[64, 64], sigma=0.1):
        super(GaussianActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_actor = FCNetwork(state_dim, action_dim, hiddens_actor, last_func=F.tanh)
        self.fc_critic = FCNetwork(state_dim, 1, hiddens_critic, last_func=F.softplus)
        self.sigma = sigma

    def forward(self, states, actions=None):
        mu = self.fc_actor(states)
        value = self.fc_critic(states).squeeze(-1)

        v_sigma = self.sigma
        dist = torch.distributions.Normal(mu, v_sigma)
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy, value

    def state_values(self, states):
        return self.fc_critic(states).squeeze(-1)
