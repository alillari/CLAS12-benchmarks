"""Train the PyG noise taggers on an exact FM HDF5 event cohort."""
import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import distributions, version
from itertools import chain, repeat
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data import Cohort, classification_edges, file_sha256, node_features
from .fm_reference import source_receipt
from .fm_reference.noise_metrics import (
    at_threshold, calibrate_thresholds, inverse_frequency_weights, score_summary,
)
from .fm_reference.track_eligibility import primary_track_eligibility_contract
from .graphs import GraphClassifier, OPERATORS
from .hdf5 import MIN4_NOISE_POLICY
from .hdf5_cohort import SPLITS, fm_cohort_spec, write_json
from .sequences import CachedSequence


def event_graph(event):
    return Data(x=torch.from_numpy(node_features(event, allow_position_only=True)),
                edge_index=torch.from_numpy(classification_edges(event.positions)),
                y=torch.from_numpy(event.labels["noise_target"]),
                coatjava_noise=torch.from_numpy(event.labels["coatjava_seg_pred"] < 0))


def run_provenance(device):
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    files = sorted(package.rglob("*.py")) + [package / "fm_reference" / "sources.json"]
    source_lock = root / "sources.lock.json"
    packages = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages.setdefault(re.sub(r"[-_.]+", "-", name).lower(), dist.version)
    receipt = {
        "source_sha256": {p.relative_to(package).as_posix(): file_sha256(p) for p in files},
        "sources_lock": json.loads(source_lock.read_text()) if source_lock.is_file() else None,
        "git_commit": os.environ.get("BENCHMARK_GIT_COMMIT"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if (root / ".git").exists() and shutil.which("git"):
        git = ["git", "--no-optional-locks", "-C", str(root)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        patch = subprocess.check_output(
            [*git, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD",
             "--", "src", "pyproject.toml", "sources.lock.json"],
            encoding="utf-8", errors="surrogateescape")
        receipt.update(git_commit=head, tracked_source_diff=patch)
    return receipt


def max_balanced_accuracy_threshold(labels, scores):
    """Exact FM campaign sweep with strict > and the lowest threshold on ties."""
    signal, noise = (np.sort(scores[labels == label]) for label in (0, 1))
    if not len(signal) or not len(noise):
        return None
    thresholds = np.unique(np.r_[-np.finfo(np.float32).eps, scores.astype(np.float64), 1.])
    tn = np.searchsorted(signal, thresholds, side="right")
    tp = len(noise) - np.searchsorted(noise, thresholds, side="right")
    objective = tn * len(noise) + tp * len(signal)
    return float(thresholds[np.argmax(objective)])


@torch.no_grad()
def validation_loss(model, loader, device, weights):
    model.eval()
    numerator, denominator = 0.0, 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        losses = F.cross_entropy(logits, batch.y, weight=weights, reduction="none")
        numerator += losses.sum(dtype=torch.float64).item()
        denominator += batch.y.numel() if weights is None else weights[batch.y].sum(dtype=torch.float64).item()
    if not denominator:
        raise ValueError("Validation cohort has no clusters")
    loss = numerator / denominator
    if not math.isfinite(loss):
        raise RuntimeError("Nonfinite pooled validation loss")
    return loss


@torch.no_grad()
def evaluate(model, loader, device, rows, output, split, calibration=None):
    model.eval()
    labels_all, scores_all, coat_all, sizes = [], [], [], []
    for batch in loader:
        labels_all.append(batch.y.numpy())
        coat_all.append(batch.coatjava_noise.numpy())
        sizes.extend(batch.ptr.diff().tolist())
        batch = batch.to(device)
        scores_all.append(model(batch.x, batch.edge_index).softmax(-1)[:, 1].cpu().numpy())
    labels = np.concatenate(labels_all).astype(np.int8)
    if not len(labels):
        raise ValueError(f"{split.capitalize()} cohort has no clusters to score")
    scores = np.concatenate(scores_all)
    coat = np.concatenate(coat_all).astype(np.int8)
    offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    metrics = score_summary(labels, scores, coat)
    if split == "validation":
        metrics["calibration"] = calibrate_thresholds(labels, scores, coat)
        threshold = max_balanced_accuracy_threshold(labels, scores)
        if threshold is not None:
            metrics["calibration"]["max_balanced_accuracy"] = at_threshold(labels, scores, threshold)
    else:
        metrics["threshold_source"] = "validation"
        metrics["operating_points"] = {}
        for name, point in calibration.items():
            result = at_threshold(labels, scores, point["threshold"])
            if "target_signal_retention" in point:
                result["target_signal_retention"] = point["target_signal_retention"]
            metrics["operating_points"][name] = result
    metrics["threshold_curve"] = [at_threshold(labels, scores, t) for t in np.linspace(0, 1, 101)]
    records = []
    for row, start, stop in zip(rows, offsets[:-1], offsets[1:], strict=True):
        y = labels[start:stop]
        predicted = scores[start:stop] > .5
        records.append({**row, "n_clusters": int(stop - start),
                        "adapter": np.bincount(2 * y + predicted, minlength=4).reshape(2, 2).tolist(),
                        "coatjava_unassigned": np.bincount(2 * y + coat[start:stop], minlength=4).reshape(2, 2).tolist()})
    np.savez_compressed(output / f"{split}_scores.npz", labels=labels, noise_score=scores,
                        coatjava_noise=coat, event_offsets=offsets)
    write_json(output / f"{split}_predictions.json", records)
    write_json(output / f"{split}_metrics.json", metrics)
    return metrics


def fit(operator, loaders, device, weights, config, output):
    torch.manual_seed(config["seed"])
    loaders["train"].generator.manual_seed(config["seed"])
    model = GraphClassifier(operator, in_channels=4, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    best_loss, history = math.inf, []
    checkpoint = output / "checkpoint.pt"
    batches = (batch for batch in chain.from_iterable(repeat(loaders["train"])) if batch.y.numel())
    for step, batch in zip(range(1, config["optimizer_steps"] + 1), batches):
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch.x, batch.edge_index), batch.y, weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., error_if_nonfinite=True)
        optimizer.step()
        if step % config["val_interval"] == 0 or step == config["optimizer_steps"]:
            val_loss = validation_loss(model, loaders["val"], device, weights)
            history.append(dict(step=step, train_batch_loss=loss.detach().item(), validation_loss=val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(dict(operator=operator, step=step, validation_loss=val_loss,
                                cohort_sha256=config["cohort_sha256"], state_dict=model.state_dict()), temporary)
                temporary.replace(checkpoint)
    write_json(output / "training_history.json", history)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    selected = GraphClassifier(operator, in_channels=4, num_classes=2).to(device)
    selected.load_state_dict(saved["state_dict"], strict=True)
    return selected, {"selected_checkpoint_step": saved["step"], "selected_validation_loss": best_loss,
                      "checkpoint_sha256": file_sha256(checkpoint), "checkpoint": checkpoint.name}


def run(data_root, fm_cohorts, plan_id, output, *, steps, operators=tuple(OPERATORS),
        batch_size=8, val_interval=1000, lr=3e-4, weight_decay=1e-4, seed=42,
        class_balance="none", device="cpu"):
    if min(steps, batch_size, val_interval) < 1 or seed < 0:
        raise ValueError("Positive budgets and a nonnegative seed are required")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Require a positive learning rate and nonnegative weight decay")
    if not operators or len(set(operators)) != len(operators) or set(operators) - OPERATORS.keys():
        raise ValueError("Select unique supported graph operators")
    if class_balance not in {"none", "inverse_frequency"}:
        raise ValueError("Unknown class balance mode")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    reference = source_receipt("noise_metrics.py", "track_eligibility.py")
    saved = Path(fm_cohorts).read_bytes()
    rows = json.loads(saved)
    spec = fm_cohort_spec(data_root, rows, plan_id, noise_policy=MIN4_NOISE_POLICY)
    with ExitStack() as stack:
        manifest = Path(stack.enter_context(TemporaryDirectory())) / "cohort.json"
        write_json(manifest, spec)
        cohort = stack.enter_context(Cohort(manifest))
        counts = np.zeros(2, dtype=np.int64)
        for index in range(len(cohort.roles["train"])):
            labels = cohort.event("train", index).labels["noise_target"]
            counts += np.bincount(labels, minlength=2)
        if not counts.sum():
            raise ValueError("Training cohort has no clusters")
        if not any(np.any(cohort.event("val", i).labels["noise_target"] == 0)
                   for i in range(len(cohort.roles["val"]))):
            raise ValueError("Validation cohort has no signal clusters under the FM four-cluster rule")
        class_weights = inverse_frequency_weights(counts) if class_balance == "inverse_frequency" else None
        weights = None if class_weights is None else torch.tensor(class_weights, device=device, dtype=torch.float32)
        provenance = run_provenance(device)
        output.mkdir(parents=True, exist_ok=False)
        (output / "cohorts.json").write_bytes(saved)
        (output / "cohort.json").write_bytes(manifest.read_bytes())
        write_json(output / "provenance.json", provenance)
        config = dict(task="noise", training_mode="graph_classifier", operators=list(operators),
                      optimizer="AdamW", optimizer_steps=steps, val_interval=val_interval, batch_size=batch_size,
                      learning_rate=lr, weight_decay=weight_decay, seed=seed, device=str(device),
                      torch_version=version("torch"), torch_geometric_version=version("torch-geometric"),
                      noise_class_balance=class_balance, training_class_counts=counts.tolist(),
                      class_weights=None if class_weights is None else class_weights.tolist(),
                      data_contract=dict(plan_id=plan_id, noise_policy=MIN4_NOISE_POLICY,
                                         track_eligibility_contract=primary_track_eligibility_contract()),
                      fm_reference=reference,
                      validation_selection="loss", threshold_source="validation", sample_unit="event",
                      cohort_sha256=cohort.sha256, fm_cohorts_sha256=hashlib.sha256(saved).hexdigest(),
                      provenance_sha256=file_sha256(output / "provenance.json"))
        write_json(output / "resolved_config.json", config)
        loaders = {}
        for role in SPLITS:
            graphs = CachedSequence(len(cohort.roles[role]), lambda i, role=role: event_graph(cohort.event(role, i)))
            loaders[role] = DataLoader(graphs, batch_size=batch_size, shuffle=role == "train",
                                       generator=torch.Generator())
        results = {}
        for operator in operators:
            folder = output / operator
            folder.mkdir()
            print(f"Training {operator} for {steps} updates", flush=True)
            model, checkpoint = fit(operator, loaders, device, weights, config, folder)
            validation = evaluate(model, loaders["val"], device, rows["validation"], folder, "validation")
            test = evaluate(model, loaders["test"], device, rows["test"], folder, "test", validation["calibration"])
            test.pop("threshold_curve")
            result = {**config, **checkpoint, **test, "operator": operator, "test_split": "test",
                      "train_samples": len(rows["train"]), "validation_samples": len(rows["validation"]),
                      "test_samples": len(rows["test"]), "trainable_parameters": sum(p.numel() for p in model.parameters())}
            write_json(folder / "summary.json", result)
            results[operator] = str(Path(operator) / "summary.json")
        summary = dict(schema="clas12_noise_graph_benchmark_v1", cohort_sha256=cohort.sha256,
                       event_counts={split: len(rows[split]) for split in SPLITS.values()}, models=results)
        write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--fm-cohorts", required=True, type=Path, help="cohorts.json from an FM noise adapter")
    parser.add_argument("--plan-id", required=True, help="Publication plan ID from the FM evaluation summary")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operators", nargs="+", choices=tuple(OPERATORS), default=tuple(OPERATORS))
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-interval", type=int, default=1000,
                        help="Validate every N updates and at the final update (default: 1000)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-balance", choices=("none", "inverse_frequency"), default="none")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
