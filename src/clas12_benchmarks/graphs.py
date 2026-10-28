"""Paper-based reconstruction of FM4NPP's semantic graph baselines.

This is our implementation, not the FM4NPP authors' unpublished adaptation.
Choices unspecified by the paper are documented in docs/PORTING.md.
"""
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv


class GraphClassifier(nn.Module):
    def __init__(self, operator, in_channels, num_classes, hidden=256):
        super().__init__()
        operators = {cls.__name__: cls for cls in (GATConv, GCNConv, GraphConv, SAGEConv)}
        if operator not in operators:
            raise ValueError(f"Unknown graph operator: {operator}")
        self.encoder = nn.Sequential(nn.Linear(in_channels, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, hidden))
        self.layers = nn.ModuleList(operators[operator](hidden, hidden) for _ in range(6))
        self.classifier = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x, edge_index):
        x = self.encoder(x)
        for layer in self.layers:
            x = torch.relu(layer(x, edge_index))
        return self.classifier(x)
