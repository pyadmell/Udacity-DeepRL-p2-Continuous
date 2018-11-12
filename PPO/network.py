"""Gaussian ActorCritic Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FCBody


class GaussianActorCriticNetwork:
    """連続分布を出力するような Actor-Critic Network モデル
    Actor としては State から連続を出すようなモデル
    Critic としては State から Value を出力するモデル
    """
    def __init__(self, state_dim=1, action_dim=1,
        hiddens_actor=[64, 64], hiddens_critic=[64, 64]):
        super(GaussianActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_actor = FCBody(state_size, hiddens_actor)
        self.fc_critic = FCBody(state_size, hiddens_critic)
        self.fc_actor_last = nn.Linear(hiddens_actor[-1], action_dim)
        self.fc_critic_last = nn.Linear(hiddens_actor[-1], 1)
        self.sigma = torch.zeros(1, action_dim)

    def forward(states, actions=None):
        z_actor = self.fc_actor(states)
        mu = self.fc_actor_last(z_actor)

        z_critic = self.fc_critic(states)
        value = self.fc_critic_last(z_critic)

        dist = torch.distributions.Normal(mu, self.sigma)
        if not actions:
            actions = dist.sample()
        log_prob = dist.logprob(actions)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return actions, log_prob, value
