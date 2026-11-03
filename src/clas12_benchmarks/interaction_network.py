"""Torch port of the pinned Exa.TrkX TF/Sonnet SegmentClassifier.

Source: vendor/exatrkx/exatrkx/src/tfgraphs/model.py. The upstream core updates
nodes BEFORE edges and concatenates the initial and current latents each round.
This is an explicit framework port, not the incomplete upstream ResAGNN file.
"""
import torch
from torch import nn

from .upstream import scatter_add


def mlp(width):
    return nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())


class SegmentClassifier(nn.Module):
    def __init__(self, input_dim=4, num_processing_steps=8):
        super().__init__()
        self.num_processing_steps = num_processing_steps
        self.node_encoder = mlp(input_dim)
        self.edge_encoder = mlp(128)
        self.node_core = mlp(384)
        self.edge_core = mlp(256)
        self.edge_output = nn.Linear(64, 1)

    def forward(self, x, edges):
        sender, receiver = edges
        nodes0 = self.node_encoder(x)
        edges0 = self.edge_encoder(torch.cat((nodes0[receiver], nodes0[sender]), dim=-1))
        nodes, values = nodes0, edges0
        outputs = []
        for _ in range(self.num_processing_steps):
            nodes_in = torch.cat((nodes0, nodes), dim=-1)
            edges_in = torch.cat((edges0, values), dim=-1)
            received = scatter_add(edges_in, receiver, dim_size=len(x))
            sent = scatter_add(edges_in, sender, dim_size=len(x))
            nodes = self.node_core(torch.cat((received, sent, nodes_in), dim=-1))
            values = self.edge_core(torch.cat((edges_in, nodes[receiver], nodes[sender]), dim=-1))
            outputs.append(self.edge_output(values).squeeze(-1))
        return outputs
