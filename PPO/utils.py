import torch
import torch.nn as nn
import torch.nn.functional as F


def to_tensor(self, np_array):
    return torch(np_array).float().to(self.device)


class FCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens,
                 func=F.relu, last_func=None):
        super(FCNetwork, self).__init__()

        self.func =  func
        self.last_func = last_func

        # Input Layer
        fc_first = nn.Linear(input_dim, hiddens[0])
        self.layers = nn.ModuleList([fc_first])
        # Hidden Layers
        layer_sizes = zip(hiddens[:-1], hiddens[1:])
        self.layers.extend([nn.Linear(h1, h2)
                            for h1, h2 in layer_sizes])
        # Output Layers
        self.layers.append(nn.Linear(hiddens[-1], output_dim))

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.func(layer(x))

        if self.last_func is None:
            x = self.layers[-1](x)
        else:
            x = self.last_func(self.layers[-1](x))

        return x
