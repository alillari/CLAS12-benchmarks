"""Check adapted tracking math against pinned source and independent graph ops.

Writes optional verification output externally. Requires the tracking extra;
TensorFlow/Sonnet GNN parity is a separate check_exatrkx_tf_port.py command.
This verifies defined kernels/adaptations, not equivalence of training datasets,
sampling, optimizer recipes or published physics performance.
"""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist
import scipy as sp
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.nn import functional as F

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.tracking import HitBatch, components, edge_loss, embedding_loss
from clas12_benchmarks.upstream import ROOT, definition, eggnet, exatrkx_embedding, exatrkx_filter


def method_assignments(path, class_name, method_name, names, scope):
    """Execute named, unmodified assignments from a pinned method, not its trainer."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    # The non-PID filter loss is in the else branch; select the last assignment
    # for a repeated target. Source edits that remove a required target fail.
    assignments = {n.targets[0].id: n for n in ast.walk(method)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    body = [assignments[name] for name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope[names[-1]]


def close_gradients(actual, expected, inputs, parameters=()):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    coefficients = torch.linspace(.2, 1.2, actual.numel(), device=actual.device).reshape(actual.shape)
    ga = torch.autograd.grad((actual * coefficients).sum(), inputs[0], retain_graph=True)[0]
    gb = torch.autograd.grad((expected * coefficients).sum(), inputs[1], retain_graph=True)[0]
    torch.testing.assert_close(ga, gb, rtol=3e-4, atol=3e-6)
    error = float((ga - gb).abs().max())
    if parameters:
        left, right = parameters
        ag = torch.autograd.grad((actual * coefficients).sum(), left, allow_unused=True)
        bg = torch.autograd.grad((expected * coefficients).sum(), right, allow_unused=True)
        for a, b in zip(ag, bg, strict=True):
            if (a is None) != (b is None):
                raise AssertionError("Parameter gradient presence differs")
            if a is not None:
                torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
                error = max(error, float((a - b).abs().max()))
    return {"max_output_error": float((actual - expected).detach().abs().max()), "max_gradient_error": error}


def reference_sum(src, index, dim=0, dim_size=None):
    n = dim_size if dim_size is not None else (int(index.max()) + 1 if len(index) else 0)
    incidence = F.one_hot(index, num_classes=n).to(src.dtype).T
    return incidence @ src


def reference_softmax(src, index):
    result = torch.empty_like(src)
    for label in torch.unique(index):
        mask = index == label
        result[mask] = src[mask].softmax(dim=0)
    return result


class ReferenceKNN:
    def get_graph(self, batch, k, r=None, time_yes=False):
        x = batch.hit_embedding.detach().cpu().numpy()
        distances = cdist(x, x)
        np.fill_diagonal(distances, np.inf)
        nearest = np.argsort(distances, axis=1)[:, :min(k, len(x) - 1)]
        target = np.broadcast_to(np.arange(len(x))[:, None], nearest.shape)
        keep = np.ones_like(nearest, dtype=bool) if r is None else distances[target, nearest] <= r
        return torch.as_tensor(np.stack((nearest[keep], target[keep])), device=batch.hit_embedding.device)


def check_eggnet(config, device):
    make_mlp = definition("vendor/eggnet/eggnet/models/utils/utils.py", "make_mlp", {})
    ref_class = definition("vendor/eggnet/eggnet/models/eggnet.py", "EggNet", {
        "F": F, "make_mlp": make_mlp, "scatter_add": reference_sum, "softmax": reference_softmax,
        "nearest_neighboring": SimpleNamespace(reference=ReferenceKNN)})
    actual = eggnet(config).to(device)
    reference = ref_class(dict(config, knn_algorithm="reference", checkpoint=False)).to(device)
    reference.load_state_dict(actual.state_dict(), strict=True)
    x = torch.randn(17, 12, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    wrap = lambda z: HitBatch(**{f"hit_f{i}": z[:, i] for i in range(12)})
    return close_gradients(actual(wrap(x)), reference(wrap(y)), (x, y),
                           (list(actual.parameters()), list(reference.parameters())))


def check_exatrkx_forwards(config, device):
    results = {}
    for name, model in (("embedding", exatrkx_embedding(config["embedding"])),
                        ("filter", exatrkx_filter(config["filter"]))):
        model = model.to(device)
        x = torch.randn(13, 4, device=device, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        if name == "embedding":
            expected = y
            for layer in model.layers:
                expected = torch.tanh(F.linear(expected, layer.weight, layer.bias))
            expected = F.linear(expected, model.emb_layer.weight, model.emb_layer.bias)
            actual = model(x)
        else:
            edges = torch.tensor([[0, 2, 4, 7, 10], [1, 5, 8, 11, 12]], device=device)
            expected = F.linear(torch.cat((y[edges[0]], y[edges[1]]), dim=-1),
                                model.input_layer.weight, model.input_layer.bias)
            for layer in model.layers:
                expected = torch.tanh(F.linear(expected, layer.weight, layer.bias))
                if config["filter"]["layernorm"]:
                    expected = F.layer_norm(expected, model.layernorm.normalized_shape,
                                            model.layernorm.weight, model.layernorm.bias, model.layernorm.eps)
                if config["filter"]["batchnorm"]:
                    raise ValueError("This independent check targets the supplied non-BatchNorm recipes")
            expected = F.linear(expected, model.output_layer.weight, model.output_layer.bias)
            actual = model(x, edges)
        results[name] = close_gradients(actual, expected, (x, y),
                                       (list(model.parameters()), list(model.parameters())))
    return results


def check_exatrkx_losses(device):
    z = torch.tensor([[0., .1], [.2, .3], [.6, -.1], [.1, .2], [1.5, -.5]], device=device, requires_grad=True)
    labels = torch.tensor([0, 0, 1, -1, 1], device=device)
    pairs = torch.triu_indices(len(z), len(z), offset=1, device=device)
    truth = (labels[pairs[0]] >= 0) & (labels[pairs[0]] == labels[pairs[1]])
    terms = []
    for mask, target in ((truth, 1.), (~truth, -1.)):
        edges = pairs[:, mask]
        terms.append(method_assignments("vendor/exatrkx/exatrkx/src/embedding/embedding_base.py",
            "EmbeddingBase", "training_step", ["reference", "neighbors", "d", "loss"],
            {"torch": torch, "spatial": z, "e_spatial": edges,
             "hinge": torch.full((edges.shape[1],), target, device=device),
             "self": SimpleNamespace(hparams={"margin": 1.})}))
    actual = embedding_loss(z, labels)
    expected = torch.stack(terms).mean()
    kernel = close_gradients(actual, expected, (z, z))
    edge_results = {}
    for case, truth_values in (("mixed", [1, 0, 0, 1, 0]), ("positive", [1] * 5), ("negative", [0] * 5)):
        logits = torch.tensor([-.7, .2, 1.1, -.1, .4], device=device, requires_grad=True)
        edge_truth = torch.tensor(truth_values, dtype=torch.bool, device=device)
        terms, weights = [], []
        for mask, weight in ((edge_truth, 1.25), (~edge_truth, 1.)):
            if mask.any():
                reference = method_assignments("vendor/exatrkx/exatrkx/src/filter/filter_base.py",
                    "FilterBaseBalanced", "training_step", ["loss"], {"F": F, "output": logits[mask],
                    "batch": SimpleNamespace(y=edge_truth[mask].float()),
                    "combined_indices": torch.arange(int(mask.sum()), device=device),
                    "weight": torch.tensor(1., device=device)})
                terms.append(weight * reference)
                weights.append(weight)
        edge_results[case] = close_gradients(edge_loss(logits, edge_truth, 1.25),
                                             torch.stack(terms).sum() / sum(weights), (logits, logits))
    return {"embedding_source_kernel_with_explicit_class_means": kernel,
            "filter_source_bce_with_explicit_class_means": edge_results}


def check_components():
    prepare = definition("vendor/exatrkx/exatrkx/scripts/tracks_from_gnn.py", "prepare", {"sp": sp, "np": np})
    edges = np.array([[0, 1, 3], [1, 2, 4]])
    matrix = prepare(np.array([.9, .8, .95]), *edges, 7).tocsr()
    reference = DBSCAN(eps=1., min_samples=1, metric="precomputed").fit_predict(matrix)
    counts = np.bincount(reference)
    reference = np.where(counts[reference] >= 2, reference, -1)
    # The upstream sparse reconstruction omits trailing unused hits. Our output
    # contract retains them as background instead of shortening the event.
    reference = np.pad(reference, (0, 7 - len(reference)), constant_values=-1)
    actual = components(7, edges, min_size=2)
    relation = lambda x: (x[:, None] == x[None, :]) & (x[:, None] >= 0) & (x[None, :] >= 0)
    np.testing.assert_array_equal(relation(actual), relation(reference))
    return {"passed": True, "policy": "same retained edges; all retained distances accepted; singleton/unused nodes background"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/tracking.json"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(2)
    # The independent incidence-matrix sum must not quantize messages to TF32
    # while the production index-add sum retains FP32. Verify the same precision.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(72)
    report = {"passed": True, "device": args.device,
              "config_sha256": file_sha256(args.config), "torch": torch.__version__,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "eggnet_forward_and_gradients": check_eggnet(config["eggnet"], args.device),
              "exatrkx_forwards": check_exatrkx_forwards(config["exatrkx"], args.device),
              "exatrkx_loss_kernels": check_exatrkx_losses(args.device), "track_partition": check_components()}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
