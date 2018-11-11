"""Gaussian ActorCritic Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork:
    def __init__(self):
        pass

    def forward(states, actions=None):
        pass

    def save(self, fpath):
        pass

    def restore(self, fpath):
        pass


class RandomContinuousNetwork(BaseNetwork):
    def __init__(self):
        pass

    def forward(states, actions=None):
        pass


class GaussianActorCriticNetwork(BaseNetwork):
    """連続分布を出力するような Actor-Critic Network モデル
    Actor としては State から連続を出すようなモデル
    Critic としては State から Value を出力するモデル
    """
    def __init__(self, state_size=1, action_size=1,
        hiddens_phi=[128, 64, 32],
        hiddens_actor=[],
        hiddens_critic=[]):
        super(GaussianActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        fc_first = nn.Linear(state, hiddens_phi[0])
        self.network_phi = nn.ModuleList([fc_first])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.network_phi.extend([nn.Linear(h1, h2)
                                   for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)


        model_phi = ...
        # actorモデルの生成
        model_actor = ...
        # criticモデルの生成
        model_critic = ...

    def forward(states, actions=None):
        # phi予測
        # actor モデル
        # critic モデル
        mu =
        if not actions is None:
            # action draw
            actions = dist.sample()
        logprob = dist.logprob(actions)
