"""Train and compare supervised CVT cluster noise taggers."""
import argparse
import hashlib
from itertools import count
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .data import classification_edges, file_sha256
from .distance_graphs import DistanceGraphClassifier, DistanceGraphConfig
from .fm_reference.noise_metrics import inverse_frequency_weights, score_summary
from .graph_noise_metrics import (
    RETENTION_TARGETS, bootstrap_rates, calibration, event_confusions, exact_curves,
    grouped_results, operating_points,
)
from .hdf5 import MIN4_NOISE_POLICY
from .hdf5_cohort import write_json
from .provenance import run_provenance


class Split:
    def __init__(self, root, receipt, role):
        path = root / f"{role}.npz"
        if file_sha256(path) != receipt["splits"][role]["sha256"]:
            raise ValueError(f"Cached {role} cohort changed")
        with np.load(path) as data:
            self.rows = dict(data)
        self.offsets = self.rows["offsets"]
        self.sizes = np.diff(self.offsets)
        self.edges = {}

    def frozen_features(self, root, role, events):
        path = root / f"{role}.npy"
        info = json.loads((root / "features.json").read_text())["splits"][role]
        if not events <= info["events"] or file_sha256(path) != info["sha256"]:
            raise ValueError(f"Invalid frozen FM feature cache for {role}")
        result = np.load(path, mmap_mode="r")
        if result.shape != (self.offsets[info["events"]], 512) or result.dtype != np.float32:
            raise ValueError(f"Frozen FM feature cache shape or dtype changed for {role}")
        return result[:self.offsets[events]]

    def features(self, sizes):
        end = self.offsets[sizes] if isinstance(sizes, int) else self.offsets[-1]
        x, y, z = self.rows["xyz"][:end].T
        radius = np.hypot(x, y)
        eta = np.arcsinh(z / radius)
        return np.column_stack((eta, x / radius, y / radius, radius))

    def batch(self, members, features, device, sage):
        sizes = self.sizes[members]
        max_size = int(sizes.max())
        mask = np.arange(max_size) < sizes[:, None]
        points = np.zeros((*mask.shape, features.shape[1]), dtype=np.float32)
        targets = np.full(mask.shape, -100, dtype=np.int64)
        edge_cols, degrees, increment = [], [], 0
        for row, member in enumerate(members):
            start, stop = self.offsets[member:member + 2]
            size = stop - start
            values = features[start:stop]
            points[row, :size] = values
            targets[row, :size] = self.rows["labels"][start:stop]
            if sage:
                if member not in self.edges:
                    src, dst = classification_edges(values)
                    self.edges[member] = src.astype(np.int32), np.bincount(dst, minlength=size)
                cols, degree = self.edges[member]
                edge_cols.append(cols + increment)
                degrees.append(degree)
                increment += size
        if sage:
            degree = np.concatenate(degrees)
            columns = np.concatenate(edge_cols, dtype=np.int64)
            edge = torch.sparse_csr_tensor(
                torch.from_numpy(np.r_[0, degree.cumsum()]), torch.from_numpy(columns),
                torch.ones(len(columns)), size=(increment, increment)).to(device)
            return torch.from_numpy(points[mask]).to(device), edge, torch.from_numpy(targets[mask]).to(device)
        return tuple(torch.from_numpy(x).to(device) for x in (points, mask, targets))


def normalized(raw, scales):
    return (raw / np.asarray((scales["eta"], 1, 1, scales["radius"]), dtype=np.float32)).astype(np.float32)


def train_batches(split, limit, batch_size, seed):
    rng = np.random.default_rng(seed)
    for _ in count():
        order = rng.permutation(limit)
        batches = []
        # Group similar occupancies locally to limit padding while preserving all events.
        bucket = 8 * batch_size
        for start in range(0, limit, bucket):
            members = order[start:start + bucket]
            members = members[np.argsort(split.sizes[members], kind="stable")]
            batches.extend(members[i:i + batch_size] for i in range(0, len(members), batch_size))
        for index in rng.permutation(len(batches)):
            members = batches[index]
            if split.sizes[members].sum():
                yield members


def make_model(arm, scales, cached=False):
    choices = arm["model"]
    if choices["architecture"] == "sage":
        from .graphs import GraphClassifier

        return GraphClassifier("SAGEConv", in_channels=4, num_classes=2, hidden=choices.get("width", 256))
    if choices["architecture"] in {"fm_adapter", "fm_frozen"}:
        from .graph_noise_fm import FMNoiseClassifier

        checkpoint = None if cached else choices.get("checkpoint")
        return FMNoiseClassifier(scales, checkpoint, choices.get("sha256"), cached=cached)
    return DistanceGraphClassifier(4, DistanceGraphConfig(**choices))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict(model, data, features, limit, batch_size, device, sage, weights=None):
    model.eval()
    scores, total, denom = [], 0., 0.
    for i in range(0, limit, batch_size):
        members = np.arange(i, min(limit, i + batch_size))
        if not data.sizes[members].sum():
            continue
        x, relation, y = data.batch(members, features, device, sage)
        logits = model(x, relation)
        masked = y != -100
        real_labels, real_logits = y[masked], logits[masked]
        if weights is not None:
            total += F.cross_entropy(real_logits, real_labels, weight=weights, reduction="sum").item()
            denom += weights[real_labels].sum().item()
        else:
            total += F.cross_entropy(real_logits, real_labels, reduction="sum").item()
            denom += real_labels.numel()
        scores.append(real_logits.softmax(-1)[:, 1].cpu().numpy())
    if not denom or not math.isfinite(total):
        raise RuntimeError("Empty or nonfinite evaluation")
    return np.concatenate(scores), total / denom


def fit(arm, train, val, settings, device, folder, cohort_sha, fm_cache=None):
    budget, validation = settings["train"], settings["validation"]
    steps, val_interval, batch_size = (settings[k] for k in ("steps", "val_interval", "batch_size"))
    if min(steps, val_interval, batch_size) < 1:
        raise ValueError("Updates, validation cadence and batch size must be positive")
    if not 1 <= budget <= len(train.sizes) or not 1 <= validation <= len(val.sizes):
        raise ValueError("Requested split sizes exceed cache or are empty")
    seed = settings["seed"]
    torch.manual_seed(seed)
    sage = arm["model"]["architecture"] == "sage"
    raw = train.features(budget)
    scales = dict(eta=max(1e-6, float(np.abs(raw[:, 0]).max())), radius=float(raw[:, 3].max()))
    if fm_cache is None:
        tx, vx = normalized(raw, scales), normalized(val.features(validation), scales)
    else:
        tx = train.frozen_features(fm_cache, "train", budget)
        vx = val.frozen_features(fm_cache, "val", validation)
    training_y = train.rows["labels"][:train.offsets[budget]]
    val_y = val.rows["labels"][:val.offsets[validation]]
    val_coat = val.rows["coatjava_noise"][:len(val_y)]
    counts = np.bincount(training_y, minlength=2)
    weights = (torch.tensor(inverse_frequency_weights(counts), device=device, dtype=torch.float32)
               if arm["class_balance"] == "inverse_frequency" else None)
    if arm["class_balance"] not in {"none", "inverse_frequency"}:
        raise ValueError("Unknown class balance")
    model = make_model(arm, scales, cached=fm_cache is not None).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=arm["lr"], weight_decay=1e-4)
    warmup = min(100, max(1, steps // 10))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .05 + .95 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best_loss, train_seconds, n_events, n_points = [], math.inf, 0., 0, 0
    checkpoint = folder / "checkpoint.pt"
    sync(device)
    start = perf_counter()
    batches = train_batches(train, budget, batch_size, seed)
    for step in range(1, steps + 1):
        members = next(batches)
        x, relation, y = train.batch(members, tx, device, sage)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, relation)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.flatten(), weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.)
        optimizer.step()
        scheduler.step()
        n_events += len(members)
        n_points += int(train.sizes[members].sum())
        if step % val_interval == 0 or step == steps:
            sync(device)
            train_seconds += perf_counter() - start
            train_loss = float(loss.detach())
            if not math.isfinite(train_loss):
                raise RuntimeError(f"Nonfinite train loss at step {step}")
            scores, loss_val = predict(model, val, vx, validation, batch_size, device, sage, weights)
            points = operating_points(val_y, scores, calibration(val_y, scores, val_coat))
            objective = float(np.mean([points[name]["noise_rejection"] for name in RETENTION_TARGETS]))
            entry = dict(step=step, train_batch_loss=train_loss, validation_loss=loss_val,
                         validation_mean_rejection=objective)
            history.append(entry)
            print(json.dumps(dict(arm=arm["name"], **entry)), flush=True)
            if loss_val < best_loss:
                best_loss = loss_val
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(dict(arm=arm, settings=settings, scales=scales, cohort_sha256=cohort_sha,
                                step=step, state_dict={k: v for k, v in model.state_dict().items()
                                                       if not k.startswith("backbone.")}), temporary)
                temporary.replace(checkpoint)
                best = dict(entry, validation_operating_points=points)
            model.train()
            start = perf_counter()
    saved = torch.load(checkpoint, weights_only=True, map_location=device)
    missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise ValueError("Selected checkpoint does not match the noise model")
    write_json(folder / "training_history.json", history)
    result = dict(**best, arm=arm, settings=settings, normalization=scales,
                  training_class_counts=counts.tolist(),
                  class_weights=None if weights is None else weights.tolist(),
                  trainable_parameters=sum(p.numel() for p in trainable),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  native_fm_head_sha256=getattr(model, "source", None),
                  fm_feature_cache=fm_cache is not None,
                  training_seconds=train_seconds, training_events_per_s=n_events / train_seconds,
                  training_clusters_per_s=n_points / train_seconds,
                  train_val_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
                  if device.type == "cuda" else None,
                  checkpoint_sha256=file_sha256(checkpoint))
    write_json(folder / "validation.json", result)
    return model, result


@torch.inference_mode()
def benchmark_forward(model, data, features, batch_size, device, sage, warm_each_batch=False):
    model.eval()
    out = {}
    rng = np.random.default_rng(62)
    order = rng.permutation(len(data.sizes))[:256]
    largest = np.argsort(data.sizes)[-min(32, len(data.sizes)):]
    for name, rows, size in (("batch", order, batch_size), ("single", order[:128], 1),
                             ("largest_single", largest, 1)):
        durations, hits, events = [], 0, 0
        for start in range(0, len(rows), size):
            items = rows[start:start + size]
            x, edges, _ = data.batch(items, features, device, sage)
            if not data.sizes[items].sum():
                continue
            if warm_each_batch or not durations:
                model(x, edges)
            sync(device)
            tick = perf_counter()
            model(x, edges)
            sync(device)
            durations.append(perf_counter() - tick)
            hits += int(data.sizes[items].sum())
            events += len(items)
        out[name] = dict(batches=len(durations), events=events,
                         events_per_s=events / sum(durations), clusters_per_s=hits / sum(durations),
                         latency_ms_p50=float(np.median(durations) * 1000),
                         latency_ms_p95=float(np.quantile(durations, .95) * 1000),
                         max_occupancy=int(data.sizes[rows].max()))
    return out


def evaluate(model, result, data, output, device, fm_cache=None):
    scores_features = (normalized(data.features(None), result["normalization"]) if fm_cache is None
                       else data.frozen_features(fm_cache, "test", len(data.sizes)))
    batch_size = result["settings"]["batch_size"]
    sage = result["arm"]["model"]["architecture"] == "sage"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scores, _ = predict(model, data, scores_features, len(data.sizes), batch_size, device, sage)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    labels, coat, detector, layer = (data.rows[key] for key in ("labels", "coatjava_noise", "detector", "layer"))
    points = {name: {key: value for key, value in point.items()
                     if key in {"threshold", "target_signal_retention", "target_noise_rejection"}}
              for name, point in result["validation_operating_points"].items()}
    curves = exact_curves(labels, scores)
    summary = score_summary(labels, scores, coat)
    summary["noise_average_precision"] = curves.pop("noise_average_precision")
    summary.update(operating_points=operating_points(labels, scores, points), threshold_source="validation",
                   groups=grouped_results(labels, scores, coat, points, data.offsets, detector, layer),
                   occupancy_quantiles=np.quantile(data.sizes, [0, .5, .9, .95, .99, 1]).tolist(),
                   inference=benchmark_forward(model, data, scores_features, batch_size, device, sage),
                   forward_peak_gpu_allocated_bytes=peak)
    np.savez_compressed(output / "test_curves.npz", **curves)
    np.savez_compressed(output / "test_scores.npz", noise_score=scores, event_offsets=data.offsets,
                        **{key: data.rows[key] for key in ("labels", "coatjava_noise", "detector", "layer",
                                                          "source_event_index", "source_id", "original_ordinal")})
    write_json(output / "test.json", summary)
    return summary, points, scores


def select(arms, prior):
    if prior["phase"] != "search":
        raise ValueError("Selection must come from the validation-only search")
    models = {}
    for arm in arms:
        name = arm["model"]["architecture"]
        row = prior["arms"][arm["name"]]
        if row["arm"] != arm:
            raise ValueError(f"Search configuration differs for {arm['name']}")
        val = row["validation_mean_rejection"]
        if name not in models or val > models[name][0]:
            models[name] = (val, arm)
    return [item[1] for item in models.values()]


def run(cache, config, phase, output, device, selection=None, fm_cache=None, *, selection_sha256=None):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.set_float32_matmul_precision("highest")
    settings = config[phase]
    if settings["test"] and selection is None:
        raise ValueError("Provide a validation-only selection receipt before test")
    search_arms = config["arms"]
    if selection:
        if selection["cohort_sha256"] != file_sha256(cache / "cohort.json"):
            raise ValueError("Validation selection came from a different cohort")
        arms = select(search_arms, selection) + settings.get("comparators", [])
    else:
        arms = [arm for arm in search_arms if arm["name"] in settings["arms"]]
    if not arms:
        raise ValueError("No arms selected")
    receipt = json.loads((cache / "cohort.json").read_text())
    if receipt["noise_policy"] != MIN4_NOISE_POLICY:
        raise ValueError("Unexpected noise policy")
    cohort_sha = file_sha256(cache / "cohort.json")
    if fm_cache is not None:
        fm_receipt = json.loads((fm_cache / "features.json").read_text())
        if fm_receipt["cohort_sha256"] != cohort_sha:
            raise ValueError("Frozen features use a different physical event roster")
        for arm in arms:
            if arm["model"]["architecture"] == "fm_frozen":
                if arm["model"]["sha256"] != fm_receipt["checkpoint_sha256"]:
                    raise ValueError("Frozen features use a different backbone checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    train, val = (Split(cache, receipt, role) for role in ("train", "val"))
    report = dict(schema="clas12_graph_noise_v1", phase=phase, settings=settings, cohort=receipt,
                  cohort_sha256=cohort_sha, platform=platform.machine(), torch_version=torch.__version__,
                  device=str(device), cuda_version=torch.version.cuda,
                  matmul_precision=torch.get_float32_matmul_precision(),
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  checkpoint_selection="minimum pooled validation loss",
                  configuration_selection="mean validation noise rejection at the four retention targets",
                  selection_receipt_sha256=selection_sha256, arms={})
    provenance = output / "provenance.json"
    provenance_record = run_provenance(device)
    write_json(provenance, provenance_record)
    report["source_sha256"] = provenance_record["source_sha256"]
    report["provenance_sha256"] = file_sha256(provenance)
    write_json(output / "resolved_config.json", dict(phase=phase, settings=settings, arms=arms))
    final = {}
    for arm in arms:
        name = arm["name"]
        cache_for_arm = fm_cache if arm["model"]["architecture"] == "fm_frozen" else None
        folder = output / name
        folder.mkdir()
        model, result = fit(arm, train, val, settings, device, folder, cohort_sha, cache_for_arm)
        report["arms"][name] = result
        if settings["test"]:
            final[name] = (model.cpu(), cache_for_arm)
        del model
    if settings["test"]:
        # All model and checkpoint decisions are frozen before opening the test archive.
        test = Split(cache, receipt, "test")
        labels, coat = test.rows["labels"], test.rows["coatjava_noise"]
        paired = {name: {"COATJAVA": event_confusions(labels, coat, test.offsets)}
                  for name in RETENTION_TARGETS}
        for name, (model, cache_for_arm) in final.items():
            folder = output / name
            summary, points, scores = evaluate(model.to(device), report["arms"][name], test, folder,
                                              device, cache_for_arm)
            model.cpu()
            report["arms"][name]["test"] = {k: v for k, v in summary.items() if k != "groups"}
            for point, confusion in paired.items():
                prediction = scores > points[point]["threshold"]
                confusion[name] = event_confusions(labels, prediction, test.offsets)
        report["bootstrap"] = {name: bootstrap_rates(tables) for name, tables in paired.items()}
    write_json(output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, help="Training stage defined in the JSON configuration")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--fm-features", type=Path)
    args = parser.parse_args()
    selection = args.selection.read_bytes() if args.selection else None
    run(args.cache, json.loads(args.config.read_text()), args.phase, args.output, args.device,
        json.loads(selection) if selection is not None else None, args.fm_features,
        selection_sha256=hashlib.sha256(selection).hexdigest() if selection is not None else None)


if __name__ == "__main__":
    main()
