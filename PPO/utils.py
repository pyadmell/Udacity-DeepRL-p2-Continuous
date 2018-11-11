class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units, gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        fc_list = [nn.Linear(dim_in, dim_out)
                   for dim_in, dim_out in zip(dims[:-1], dims[1:])]
        self.layers = nn.ModuleList()
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x
