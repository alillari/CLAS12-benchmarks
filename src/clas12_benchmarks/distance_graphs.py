"""Independent GravNet and GarNet for masked batches of detector events."""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DistanceGraphConfig:
    architecture: str
    width: int = 48
    layers: int = 3
    propagate: int = 16
    coordinates: int = 4
    neighbors: int = 16
    aggregators: int = 4
    distance_scale: float = 1.0
    query_chunk: int = 128

    def __post_init__(self):
        if self.architecture not in {"gravnet", "garnet"}:
            raise ValueError(f"Unknown distance-graph architecture: {self.architecture}")
        dimensions = (self.width, self.layers, self.propagate, self.coordinates,
                      self.neighbors, self.aggregators, self.query_chunk)
        if any(type(size) is not int or size < 1 for size in dimensions):
            raise ValueError("Graph dimensions must be positive integers")
        if not math.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("Distance scale must be positive and finite")


def canonical_order(x, mask):
    """Lexicographically order observed features so ties do not depend on input order."""
    with torch.no_grad():
        order = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        for column in [*x.unbind(dim=-1)[::-1], ~mask]:
            order = order.gather(1, column.gather(1, order).argsort(dim=1, stable=True))
    return order


class GravNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.space = nn.Linear(config.width, config.coordinates, bias=False)
        self.propagate = nn.Linear(config.width, config.propagate)
        self.out = nn.Linear(config.width + 2 * config.propagate, config.width)
        self.k = config.neighbors
        self.scale = config.distance_scale
        self.chunk = config.query_chunk

    def forward(self, x, mask):
        n = x.shape[1]
        if n < 2:
            zeros = x.new_zeros((*x.shape[:2], 2 * self.propagate.out_features))
            return self.out(torch.cat((x, zeros), dim=-1)).tanh()
        # Bias would be a global translation and could not learn from relative distances.
        latent = self.space(x)
        features = self.propagate(x)
        k = min(self.k, n - 1)
        rows = torch.arange(n, device=x.device)
        batches = torch.arange(x.shape[0], device=x.device)[:, None, None]
        pooled = []
        for start in range(0, n, self.chunk):
            query = rows[start:start + self.chunk]
            centers = latent[:, start:start + self.chunk]
            with torch.no_grad():
                d = torch.cdist(centers, latent, compute_mode="donot_use_mm_for_euclid_dist")
                d.masked_fill_(~mask[:, None, :] | (query[:, None] == rows), torch.inf)
                nearest = d.argsort(dim=-1, stable=True)[..., :k]
                valid = d.gather(-1, nearest).isfinite().unsqueeze(-1)
            # Discrete neighbor selection is detached. The Gaussian retains its gradient.
            distance2 = (latent[batches, nearest] - centers.unsqueeze(-2)).square().sum(-1)
            weight = torch.exp(-self.scale * distance2).unsqueeze(-1)
            messages = features[batches, nearest] * weight
            mean = messages.masked_fill(~valid, 0).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1)
            maximum = messages.masked_fill(~valid, -torch.inf).amax(dim=-2)
            maximum = maximum.masked_fill(~valid.any(dim=-2), 0)
            pooled.append(torch.cat((mean, maximum), dim=-1))
        return self.out(torch.cat((x, torch.cat(pooled, dim=1)), dim=-1)).tanh()


class GarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distance = nn.Linear(config.width, config.aggregators)
        self.propagate = nn.Linear(config.width, config.propagate)
        self.out = nn.Linear(config.width + 2 * config.aggregators * config.propagate, config.width)
        self.scale = config.distance_scale

    def forward(self, x, mask):
        weight = torch.exp(-self.scale * self.distance(x).abs())
        messages = weight.unsqueeze(-1) * self.propagate(x).unsqueeze(-2)
        valid = mask[:, :, None, None]
        mean = messages.masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = messages.masked_fill(~valid, -torch.inf).amax(dim=1)
        maximum = maximum.masked_fill(~valid.any(dim=1), 0)
        aggregate = torch.cat((mean, maximum), dim=-1)
        returned = (weight.unsqueeze(-1) * aggregate.unsqueeze(1)).flatten(start_dim=-2)
        return self.out(torch.cat((x, returned), dim=-1)).tanh()


class DistanceGraphClassifier(nn.Module):
    """Return two logits per measurement; padded positions are zero."""
    def __init__(self, in_channels, config):
        super().__init__()
        self.config = config
        width = config.width
        self.encoder = nn.Sequential(nn.Linear(in_channels, width), nn.SiLU(), nn.Linear(width, width))
        operator = GravNet if config.architecture == "gravnet" else GarNet
        self.layers = nn.ModuleList(operator(config) for _ in range(config.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.layers)
        self.head = nn.Sequential(nn.Linear((config.layers + 1) * width, width), nn.SiLU(),
                                  nn.Linear(width, 2))

    def forward(self, x, mask):
        if x.shape[1] == 0:
            return x.new_zeros((*x.shape[:2], 2))
        order = None
        if self.config.architecture == "gravnet":
            order = canonical_order(x, mask)
            x = x.gather(1, order.unsqueeze(-1).expand_as(x))
            mask = mask.gather(1, order)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        x = self.encoder(x).masked_fill(~mask.unsqueeze(-1), 0)
        states = [x]
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x = norm(x + layer(x + x.sum(dim=1, keepdim=True) / count, mask))
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            states.append(x)
        logits = self.head(torch.cat(states, dim=-1)).masked_fill(~mask.unsqueeze(-1), 0)
        if order is not None:
            logits = torch.zeros_like(logits).scatter(1, order.unsqueeze(-1).expand_as(logits), logits)
        return logits
