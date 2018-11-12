import torch
import torch.nn as nn
import torch.nn.functional as F


class FCBody(nn.Module):
    def __init__(self, state_dim, hiddens, func=F.relu):
        super(FCBody, self).__init__()

        fc_first = nn.Linear(state_dim, hiddens[0])
        self.layers = nn.ModuleList([fc_first])
        layer_sizes = zip(hiddens[:-1], hiddens[1:])
        self.layers.extend([nn.Linear(h1, h2)
                            for h1, h2 in layer_sizes])

    def forward(self, x):
        for layer in self.layers:
            x = self.func(layer(x))
        return x
