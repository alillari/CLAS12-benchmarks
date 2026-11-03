"""Run pinned upstream forward definitions without their legacy training stacks.

Only named, unmodified AST definitions are compiled. EggNet uses native torch
segment reductions and exact kNN; Exa.TrkX's two MLPs use a parameter-only base
instead of Lightning. No upstream files are changed or import stubs installed.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]


def scatter_add(src, index, dim=0, dim_size=None):
    if dim != 0:
        raise ValueError("This adapter implements EggNet's dim=0 reduction only")
    n = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    return src.new_zeros((n, *src.shape[1:])).index_add_(0, index, src)


def softmax(src, index):
    n = int(index.max()) + 1 if index.numel() else 0
    expanded = index.reshape(-1, *([1] * (src.ndim - 1))).expand_as(src)
    maxima = src.new_full((n, *src.shape[1:]), -torch.inf)
    maxima.scatter_reduce_(0, expanded, src.detach(), reduce="amax", include_self=True)
    exps = (src - maxima[index]).exp()
    return exps / (scatter_add(exps, index, dim_size=n)[index] + 1e-16)


@torch.no_grad()
def neighbor_edges(x, k, radius=None, chunk_size=1024):
    """Exact Euclidean neighbor -> center edges; bounded distance workspace."""
    if k < 1:
        raise ValueError("k must be positive")
    n = len(x)
    k = min(k, n - 1)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    edges = []
    for begin in range(0, n, chunk_size):
        centers = torch.arange(begin, min(n, begin + chunk_size), device=x.device)
        d = torch.cdist(x[centers], x, compute_mode="donot_use_mm_for_euclid_dist")
        d[torch.arange(len(centers), device=x.device), centers] = torch.inf
        distances, neighbors = d.topk(k, largest=False, sorted=True)
        target = centers[:, None].expand_as(neighbors)
        keep = torch.ones_like(distances, dtype=torch.bool) if radius is None else distances <= radius
        edges.append(torch.stack((neighbors[keep], target[keep])))
    return torch.cat(edges, dim=1)


class TorchKNN:
    def get_graph(self, batch, k, r=None, time_yes=False):
        return neighbor_edges(batch.hit_embedding.detach(), k, r)


class ParameterBase(nn.Module):
    """The only base behavior required by the legacy MLP constructors."""
    def __init__(self, hparams):
        super().__init__()
        self.hparams = dict(hparams)


def definition(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected exactly one {name} in {path}")
    scope = {"__name__": __name__, "torch": torch, "nn": nn, "Linear": nn.Linear, **namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def eggnet(hparams):
    mlp = definition("vendor/eggnet/eggnet/models/utils/utils.py", "make_mlp", {})
    cls = definition("vendor/eggnet/eggnet/models/eggnet.py", "EggNet", {
        "F": F, "checkpoint": checkpoint, "make_mlp": mlp,
        "scatter_add": scatter_add, "softmax": softmax,
        "nearest_neighboring": SimpleNamespace(torch_knn=TorchKNN),
    })
    if hparams.get("node_filter") or hparams.get("output_node_score"):
        raise ValueError("This runtime supports the full-node EggNet embedding path only")
    return cls(dict(hparams, knn_algorithm="torch_knn"))


def exatrkx_embedding(hparams):
    cls = definition("vendor/exatrkx/exatrkx/src/embedding/layerless_embedding.py",
                     "LayerlessEmbedding", {"EmbeddingBase": ParameterBase})
    return cls(hparams)


def exatrkx_filter(hparams):
    cls = definition("vendor/exatrkx/exatrkx/src/filter/vanilla_filter.py",
                     "VanillaFilter", {"FilterBaseBalanced": ParameterBase})
    return cls(hparams)
