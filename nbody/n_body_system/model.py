import torch
from torch import nn
from models import GCL


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, act_fn=nn.SiLU(), n_layers=4):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dimension_reduce = nn.ModuleList()
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i,
                            GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=2, act_fn=act_fn))
        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                                     act_fn,
                                     nn.Linear(hidden_nf, 3))

        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))

    def forward(self, h, edges, edge_attr=None):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        h = self.decoder(h)
        return h
