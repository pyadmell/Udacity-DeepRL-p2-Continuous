import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens, func=F.relu):
        super(FCNetwork, self).__init__()

        self.func =  func

        # Input Layer
        fc_first = nn.Linear(input_dim, hiddens[0])
        self.layers = nn.ModuleList([fc_first])
        # Hidden Layers
        layer_sizes = zip(hiddens[:-1], hiddens[1:])
        self.layers.extend([nn.Linear(h1, h2)
                            for h1, h2 in layer_sizes])
        # Output Layers
        self.layers.append(nn.Linear(hiddens[-1], output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.func(layer(x))
        x = self.layers[-1](x)
        return x
