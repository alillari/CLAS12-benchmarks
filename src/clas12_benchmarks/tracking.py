"""First CLAS12 tracking runs: real upstream forwards, shared native metrics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.nn import functional as F

from .data import Cohort, eggnet_features, file_sha256
from .sequences import CachedSequence, event_sequence, derived_sequence
from .interaction_network import SegmentClassifier
from .predictions import save_predictions
from .prepare_cohort import write_json
from .upstream import ROOT, eggnet, exatrkx_embedding, exatrkx_filter, neighbor_edges


class HitBatch(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


def validate_recipe(config):
    """Resolve new-run defaults explicitly; reject silent objective fallbacks."""
    objective = config.setdefault("eggnet_objective", "upstream_contrastive")
    if objective not in {"upstream_contrastive", "class_balanced_squared_hinge"}:
        raise ValueError(f"Unknown EggNet objective: {objective}")
    if objective == "upstream_contrastive":
        missing = {"knn_loss", "random_pairs_per_node"} - config["eggnet"].keys()
        if missing:
            raise ValueError(f"Corrected EggNet objective requires explicit sampling parameters: {sorted(missing)}")
    orientation = config["exatrkx"].setdefault("edge_orientation", "geometry")
    if orientation not in {"geometry", "legacy_index"}:
        raise ValueError(f"Unknown Exa.TrkX edge orientation: {orientation}")
    if config["exatrkx"]["filter"].get("emb_channels", 0) != 0:
        raise ValueError("The CLAS12 runner supports coordinate-only filters (emb_channels=0)")


def egg_features(event):
    if len(event.positions) >= 3:
        return eggnet_features(event)
    # Keep complete tiny events. Repeat the sole other point (or self for N=1).
    # This explicit adaptation is only needed when two distinct neighbors don't exist.
    if not len(event.positions):
        raise ValueError("Empty event")
    other = event.positions[::-1]
    return np.concatenate((event.positions, other, other), axis=1)


def embedding_output(model, event, method, device):
    if method == "eggnet":
        x = torch.as_tensor(egg_features(event), device=device)
        batch = HitBatch(**{f"hit_f{i}": x[:, i] for i in range(12)})
        return model(batch)
    return model(torch.as_tensor(event.positions, device=device))


def pair_labels(labels, edges):
    a, b = labels[edges[0]], labels[edges[1]]
    return (a >= 0) & (a == b)


def embedding_loss(z, labels, margin=1., max_pairs=2000000):
    n = len(z)
    if n * (n - 1) // 2 > max_pairs:
        raise ValueError("Event exceeds explicit all-pairs limit; no point truncation")
    edges = torch.triu_indices(n, n, offset=1, device=z.device)
    if not edges.numel():
        return z.sum() * 0
    same = pair_labels(labels, edges)
    d2 = (z[edges[0]] - z[edges[1]]).square().sum(-1)
    # Equal class means: full all-same-track positives plus every negative pair.
    # All background-background pairs are negatives; no background instance.
    terms = []
    if same.any():
        terms.append(d2[same].mean())
    if (~same).any():
        terms.append(F.relu(margin - d2[~same]).mean())
    return torch.stack(terms).mean()


def eggnet_contrastive_loss(z, labels, margin=1., max_pairs=2000000, k=10, random_pairs_per_node=10):
    """Upstream Euclidean squared hinge, with signal/kNN/random term means.

    CLAS12 adaptation: exclude self pairs and never label background positive.
    Random budget scales with event size rather than the TrackML hit count.
    """
    n = len(z)
    if n * (n - 1) // 2 > max_pairs:
        raise ValueError("Event exceeds explicit all-pairs limit; no point truncation")
    if n < 2:
        return z.sum() * 0
    pairs = torch.triu_indices(n, n, offset=1, device=z.device)
    signal = pairs[:, pair_labels(labels, pairs)]
    hard = neighbor_edges(z.detach(), k)
    random_pairs = torch.randint(n, (2, n * random_pairs_per_node), device=z.device)
    random_pairs = random_pairs[:, random_pairs[0] != random_pairs[1]]
    terms = []
    for edges in (signal, hard, random_pairs):
        if not edges.numel():
            terms.append(z.sum() * 0)
            continue
        distance = ((z[edges[0]] - z[edges[1]]).square().sum(-1) + 1e-12).sqrt()
        same = pair_labels(labels, edges)
        terms.append(torch.where(same, distance, F.relu(margin - distance)).square().mean())
    return torch.stack(terms).sum()


def edge_loss(logits, truth, positive_weight=1.):
    if not logits.numel():
        return logits.sum() * 0
    terms, weights = [], []
    if truth.any():
        terms.append(positive_weight * F.softplus(-logits[truth]).mean())
        weights.append(positive_weight)
    if (~truth).any():
        terms.append(F.softplus(logits[~truth]).mean())
        weights.append(1.)
    return torch.stack(terms).sum() / sum(weights)


@torch.no_grad()
def candidates(model, event, config, device):
    z = embedding_output(model, event, "exatrkx", device)
    edges = neighbor_edges(z, config["candidate_k"], config["candidate_radius"])
    # Treat either directed proposal as an undirected candidate, without truth.
    edges = torch.sort(edges, dim=0).values
    edges = torch.unique(edges, dim=1).cpu()
    # Old checkpoint provenance did not name an orientation. Retain its exact
    # behavior here; validate_recipe explicitly resolves geometry for new runs.
    orientation = config.get("edge_orientation", "legacy_index")
    if orientation == "legacy_index":
        return edges
    if orientation != "geometry":
        raise ValueError(f"Unknown Exa.TrkX edge orientation: {orientation}")
    # The filter concatenates endpoints asymmetrically. Choose their order from
    # observed geometry, never arbitrary row indices. Radius follows the upstream
    # outward convention; xyz breaks equal-radius ties without using truth.
    xyz = event.raw_features[:, -3:]
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], np.linalg.norm(xyz, axis=1)))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    flip = torch.from_numpy(rank[edges[0]] > rank[edges[1]])
    return torch.where(flip.unsqueeze(0), edges.flip(0), edges)


def components(n, edges, min_size=2):
    edges = np.asarray(edges, dtype=np.int64)
    if not edges.size:
        return np.full(n, -1, dtype=np.int64)
    a, b = edges
    adjacency = coo_matrix((np.ones(len(a) * 2), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    counts = np.bincount(labels)
    return np.where(counts[labels] >= min_size, labels, -1).astype(np.int64)


def metric_module(source, output):
    path = output / "track_finding_metrics.py"
    shutil.copyfile(source, path)
    spec = importlib.util.spec_from_file_location("clas12_shared_track_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(events, predictions, metrics, output=None, method=None, cohort_sha=None):
    rows, baseline = [], []
    for i, (event, prediction) in enumerate(zip(events, predictions, strict=True)):
        if len(prediction) != len(event.raw_features) or not np.issubdtype(np.asarray(prediction).dtype, np.integer):
            raise ValueError("Invalid point prediction mapping")
        row = metrics.event_track_metrics(event.labels["seg_target"], prediction)
        rows.append({"identity": event.identity, **row})
        if "coatjava_seg_pred" in event.labels:
            baseline.append(metrics.event_track_metrics(event.labels["seg_target"], event.labels["coatjava_seg_pred"]))
        if output:
            save_predictions(output / "predictions" / f"event_{i:06d}.npz", event, prediction,
                event.original_point_index, method=method, cohort_sha256=cohort_sha)
    summary = metrics.summarize_event_metrics(rows)
    result = {"metrics": summary, "metric_view": "native", "matching": vars(metrics.MatchConfig()),
              "coatjava": metrics.summarize_event_metrics(baseline)}
    if baseline:
        result["model_minus_coatjava"] = metrics.compare_metric_summaries(summary, result["coatjava"])
    if output:
        write_json(output / "summary.json", result)
        with (output / "per_event_metrics.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    return result


@torch.no_grad()
def eggnet_prediction(model, event, selection, device):
    embedding = embedding_output(model, event, "eggnet", device).cpu().numpy()
    return DBSCAN(eps=selection["eps"], min_samples=selection["min_samples"]).fit_predict(embedding).astype(np.int64)


def track_f1(summary):
    m = summary["metrics"]
    den = m["n_true_tracks"] + m["n_pred_tracks"]
    return 2 * m["n_matched_tracks"] / den if den else 0.


def learning_rate_at_step(config, step):
    if config.get("learning_rate_schedule", "constant") == "constant":
        return config["learning_rate"]
    warmup = config.get("warmup_steps", 0)
    if step < warmup:
        return config["learning_rate"] * (step + 1) / warmup
    progress = (step - warmup) / max(config["steps_per_stage"] - warmup - 1, 1)
    floor = config["min_learning_rate"]
    return floor + .5 * (config["learning_rate"] - floor) * (1 + math.cos(math.pi * min(progress, 1.)))


def train_stage(model, name, events, validation, loss_fn, config, output, device, resume=False, score_fn=None):
    if not len(events) or not len(validation):
        raise ValueError("Training and validation require at least one event")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(events))
    best, best_score, start = float("inf"), -float("inf"), 0
    last_path, best_path = output / f"{name}_last.pt", output / f"{name}_best.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        best, start, order = state["best"], state["step"], state["order"]
        best_score = state.get("best_score", -float("inf"))
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    losses = []
    explicit_validation_steps = set(config.get("validation_steps", []))
    started = time.monotonic()
    for step in range(start, config["steps_per_stage"]):
        if step and step % len(events) == 0:
            order = rng.permutation(len(events))
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_step(config, step)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, events[int(order[step % len(events)])])
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite {name} loss at {step}")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(json.dumps({"stage": name, "step": step + 1, "train_loss": float(np.mean(losses[-100:])),
                "grad_norm": float(norm), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        if (step + 1) % config["validation_interval"] == 0 or step + 1 == config["steps_per_stage"] or step + 1 in explicit_validation_steps:
            model.eval()
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            # Random-pair validation must be repeatable and not perturb training RNG.
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(config["seed"] + 17)
                val = float(np.mean([float(loss_fn(model, e)) for e in validation]))
                score = float(score_fn(model, validation)) if score_fn else None
            if not np.isfinite(val):
                raise ValueError(f"Nonfinite {name} validation loss")
            if score is not None and not np.isfinite(score):
                raise ValueError(f"Nonfinite {name} selection score")
            improved = score > best_score if score is not None else val < best
            best = min(best, val)
            if score is not None:
                best_score = max(best_score, score)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step + 1, "best": best, "validation_loss": val, "stage": name,
                "order": order, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "config": config}
            payload.update(best_score=best_score, selection_score=score)
            torch.save(payload, last_path.with_suffix(".tmp"))
            last_path.with_suffix(".tmp").replace(last_path)
            if improved:
                shutil.copyfile(last_path, best_path)
            row = {"stage": name, "step": step + 1, "validation_loss": val, "best": best,
                "improved": improved, "train_loss": float(np.mean(losses)), "elapsed_seconds": time.monotonic() - started}
            row.update(selection_score=score, learning_rate=optimizer.param_groups[0]["lr"])
            with (output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            losses = []
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return {"stage": name, "selected_step": state["step"], "validation_loss": state["validation_loss"],
            "selection_score": state.get("selection_score"),
            "checkpoint_sha256": file_sha256(best_path), "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def select_eggnet(model, events, config, metrics, device):
    embeddings = derived_sequence(events, lambda i: embedding_output(model, events[i], "eggnet", device).detach().cpu().numpy())
    trials = []
    for eps in config["dbscan_eps"]:
        for min_samples in config["dbscan_min_samples"]:
            labels = (DBSCAN(eps=eps, min_samples=min_samples).fit_predict(z).astype(np.int64) for z in embeddings)
            result = evaluate(events, labels, metrics)
            trials.append({"eps": eps, "min_samples": min_samples, "track_f1": track_f1(result), **result})
    # Stable tie break: declared grid order; no test input to selection.
    return max(trials, key=lambda row: row["track_f1"]), trials


@torch.no_grad()
def filter_probabilities(model, event, edges, device):
    return model(torch.as_tensor(event.positions, device=device), edges.to(device)).flatten().sigmoid().cpu()


@torch.no_grad()
def select_filter(model, events, graphs, config, device):
    probs = derived_sequence(events, lambda i: filter_probabilities(model, events[i], graphs[i], device))
    truths = derived_sequence(events, lambda i: pair_labels(torch.as_tensor(events[i].labels["seg_target"]), graphs[i]))
    trials = []
    positives = sum(int(y.sum()) for y in truths)
    if not positives:
        raise ValueError("No positive validation candidate edges")
    for cut in config["filter_cuts"]:
        true_kept = sum(int((y & (p >= cut)).sum()) for y, p in zip(truths, probs))
        kept = sum(int((p >= cut).sum()) for p in probs)
        trials.append({"cut": cut, "edge_recall_given_candidates": true_kept / positives,
                       "edge_precision": true_kept / kept if kept else 0., "edges_kept": kept})
    eligible = [row for row in trials if row["edge_recall_given_candidates"] >= config["filter_min_edge_recall"]]
    selected = max(eligible, key=lambda row: (row["edge_precision"], row["cut"])) if eligible else max(trials, key=lambda row: row["edge_recall_given_candidates"])
    return selected, trials


@torch.no_grad()
def graph_scores(model, events, graphs, device):
    @torch.no_grad()
    def score(i):
        event, graph = events[i], graphs[i].to(device)
        bidir = torch.cat((graph, graph.flip(0)), dim=1)
        scores = model(torch.as_tensor(event.positions, device=device), bidir)[-1].sigmoid()
        return ((scores[:graph.shape[1]] + scores[graph.shape[1]:]) * .5).cpu()
    return derived_sequence(events, score)


def labels_from_scores(events, graphs, probabilities, cut, config):
    return derived_sequence(events, lambda i: components(len(events[i].raw_features),
        graphs[i][:, probabilities[i] >= cut].numpy(), config["min_component_size"]))


def graph_predictions(model, events, graphs, cut, config, device):
    return labels_from_scores(events, graphs, graph_scores(model, events, graphs, device), cut, config)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "provenance.json").exists() and not args.resume:
        raise FileExistsError("Existing run: use a fresh output or explicit --resume")
    config = json.loads(Path(args.config).read_text())
    validate_recipe(config)
    if args.smoke:
        config.update(steps_per_stage=4, validation_interval=2, checkpoint_validation_events=4)
        if "warmup_steps" in config:
            config["warmup_steps"] = 1
        config["eggnet"].update(dbscan_eps=[.1, .5, 1.], dbscan_min_samples=[2])
        config["exatrkx"].update(gnn_cuts=[.3, .7])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing silent CPU fallback")
    with Cohort(args.cohort) as cohort:
        return _run_cohort(args, output, config, device, cohort)


def _run_cohort(args, output, config, device, cohort):
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = metric_module(args.metrics, output)
    source_files = sorted(p for folder in ("src", "configs", "vendor") for p in (ROOT / folder).rglob("*")
                          if p.is_file() and p.suffix in (".py", ".json", ".yaml") and "__pycache__" not in p.parts)
    data_hashes = cohort.data_fingerprints()
    provenance = {"method": args.method, "config": config, "cohort_sha256": cohort.sha256,
        "metric_sha256": file_sha256(output / "track_finding_metrics.py"), "metrics_source": args.metrics,
        "git_commit": os.environ.get("BENCHMARK_GIT_COMMIT"), "sources_lock": json.loads((ROOT / "sources.lock.json").read_text()),
        "data_array_sha256": data_hashes,
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files},
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_selection": "validation matched-track F1 for EggNet/GNN; stage loss for Exa.TrkX embedding/filter" if config.get("track_checkpoint_selection") else "minimum event-mean validation objective on fixed validation subset",
        "threshold_selection": "full validation matched-track global F1 for clustering/GNN; filter precision subject to candidate-edge recall",
        "loss": {"eggnet": config.get("eggnet_objective", "class_balanced_squared_hinge"),
                 "exatrkx_embedding": "class_balanced_squared_hinge", "edges": "class-mean BCE with declared GNN positive weight"},
        "smoke": args.smoke, "tiny_event_eggnet": "repeat missing coordinate neighbors; preserve every point"}
    if args.resume:
        old = json.loads((output / "provenance.json").read_text())
        for key in ("method", "config", "cohort_sha256", "metric_sha256", "source_sha256", "data_array_sha256"):
            if old[key] != provenance[key]:
                raise ValueError(f"Resume provenance mismatch: {key}")
    write_json(output / "provenance.json", provenance)
    (output / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    role_events = {}
    for role in ("train", "val"):
        count = min(len(cohort.roles[role]), 16 if role == "train" else 8) if args.smoke else len(cohort.roles[role])
        role_events[role] = event_sequence(cohort, role, count, config.get("event_cache_size", 8))
    train, val = role_events["train"], role_events["val"]
    if config.get("track_checkpoint_selection"):
        checkpoint_indices = np.random.default_rng(config["seed"] + 1).choice(len(val), min(len(val), config["checkpoint_validation_events"]), replace=False).tolist()
    else:
        checkpoint_indices = list(range(min(len(val), config["checkpoint_validation_events"])))
    checkpoint_val = [val[i] for i in checkpoint_indices]
    write_json(output / "checkpoint_validation_indices.json", checkpoint_indices)
    print(json.dumps({"phase": "start", "method": args.method, "train": len(train), "val": len(val),
        "cohort_sha256": cohort.sha256, "tiny_train_events": sum(len(e.positions) < 3 for e in train), "device": str(device)}), flush=True)
    selected_stages = []
    model = eggnet(config["eggnet"]) if args.method == "eggnet" else exatrkx_embedding(config["exatrkx"]["embedding"])
    def embed_objective(m, event):
        z = embedding_output(m, event, args.method, device)
        if args.method == "eggnet" and config.get("eggnet_objective") == "upstream_contrastive":
            return eggnet_contrastive_loss(z, torch.as_tensor(event.labels["seg_target"], device=device),
                config["margin"], config["max_pairs"], config["eggnet"]["knn_loss"], config["eggnet"]["random_pairs_per_node"])
        return embedding_loss(z, torch.as_tensor(event.labels["seg_target"], device=device), config["margin"], config["max_pairs"])
    egg_score = None
    if args.method == "eggnet" and config.get("track_checkpoint_selection"):
        def egg_score(m, events):
            ec = dict(config["eggnet"], dbscan_eps=config["eggnet"]["checkpoint_dbscan_eps"],
                dbscan_min_samples=config["eggnet"]["checkpoint_dbscan_min_samples"])
            return select_eggnet(m, events, ec, metrics, device)[0]["track_f1"]
    selected_stages.append(train_stage(model, "embedding", train, checkpoint_val, embed_objective, config, output, device, args.resume, egg_score))
    if args.method == "eggnet":
        selection, trials = select_eggnet(model, val, config["eggnet"], metrics, device)
        write_json(output / "validation_selection.json", {"selected": selection, "trials": trials})
    else:
        ec = config["exatrkx"]
        graphs = {role: derived_sequence(events, lambda i, events=events: candidates(model, events[i], ec, device))
                  for role, events in role_events.items()}
        diagnostics = {}
        for role, events in role_events.items():
            total = found = 0
            for e, graph in zip(events, graphs[role], strict=True):
                y = e.labels["seg_target"]
                total += sum(int((y == label).sum()) * (int((y == label).sum()) - 1) // 2 for label in np.unique(y) if label >= 0)
                found += int(pair_labels(torch.as_tensor(y), graph).sum())
            diagnostics[role] = {"true_pairs": total, "candidate_true_pairs": found,
                "candidate_edge_recall": found / total if total else None,
                "n_candidate_edges": sum(g.shape[1] for g in graphs[role])}
        write_json(output / "candidate_diagnostics.json", diagnostics)
        filt = exatrkx_filter(ec["filter"])
        train_items = derived_sequence(train, lambda i: (train[i], graphs["train"][i]))
        checkpoint_graphs = [graphs["val"][i] for i in checkpoint_indices]
        val_items = list(zip(checkpoint_val, checkpoint_graphs))
        def filter_objective(m, item):
            event, graph = item
            graph = graph.to(device)
            logits = m(torch.as_tensor(event.positions, device=device), graph).flatten()
            return edge_loss(logits, pair_labels(torch.as_tensor(event.labels["seg_target"], device=device), graph))
        selected_stages.append(train_stage(filt, "filter", train_items, val_items, filter_objective, config, output, device, args.resume))
        filter_selection, filter_trials = select_filter(filt, val, graphs["val"], ec, device)
        write_json(output / "filter_selection.json", {"selected": filter_selection, "trials": filter_trials})
        cut = filter_selection["cut"]
        for role, events in role_events.items():
            original_graphs = graphs[role]
            graphs[role] = derived_sequence(events, lambda i, events=events, gs=original_graphs:
                gs[i][:, filter_probabilities(filt, events[i], gs[i], device) >= cut])
        gnn = SegmentClassifier(4, ec["gnn_processing_steps"])
        def gnn_objective(m, item):
            event, graph = item
            graph = graph.to(device)
            bidir = torch.cat((graph, graph.flip(0)), dim=1)
            logits = m(torch.as_tensor(event.positions, device=device), bidir)
            truth = pair_labels(torch.as_tensor(event.labels["seg_target"], device=device), bidir)
            return torch.stack([edge_loss(output_logits, truth, ec.get("gnn_positive_weight", 1.)) for output_logits in logits]).mean()
        checkpoint_graphs = [graphs["val"][i] for i in checkpoint_indices]
        gnn_score = None
        if config.get("track_checkpoint_selection"):
            def gnn_score(m, items):
                events, gs = map(list, zip(*items))
                scores = graph_scores(m, events, gs, device)
                return max(track_f1(evaluate(events, labels_from_scores(events, gs, scores, threshold, ec), metrics))
                           for threshold in ec["checkpoint_gnn_cuts"])
        selected_stages.append(train_stage(gnn, "gnn", derived_sequence(train, lambda i: (train[i], graphs["train"][i])),
            list(zip(checkpoint_val, checkpoint_graphs)), gnn_objective, config, output, device, args.resume, gnn_score))
        trials = []
        val_scores = graph_scores(gnn, val, graphs["val"], device)
        for threshold in ec["gnn_cuts"]:
            result = evaluate(val, labels_from_scores(val, graphs["val"], val_scores, threshold, ec), metrics)
            trials.append({"gnn_cut": threshold, "track_f1": track_f1(result), **result})
        selection = max(trials, key=lambda row: row["track_f1"])
        write_json(output / "validation_selection.json", {"selected": selection, "trials": trials})
    # Freeze checkpoint identities and all thresholds before opening test events.
    write_json(output / "selected_checkpoints.json", selected_stages)
    write_json(output / "frozen_selection.json", selection)
    if args.smoke:
        write_json(output / "smoke_complete.json", {"stages": selected_stages, "validation": selection, "test_evaluated": False})
        record_resources(cohort, device, output, started)
        print("SMOKE_COMPLETE: test was not evaluated", flush=True)
        return
    test = event_sequence(cohort, "test", cache_size=config.get("event_cache_size", 8))
    with torch.no_grad():
        if args.method == "eggnet":
            predictions = derived_sequence(test, lambda i: eggnet_prediction(model, test[i], selection, device))
        else:
            raw_graphs = derived_sequence(test, lambda i: candidates(model, test[i], ec, device))
            test_graphs = derived_sequence(test, lambda i: raw_graphs[i][:,
                filter_probabilities(filt, test[i], raw_graphs[i], device) >= cut])
            predictions = graph_predictions(gnn, test, test_graphs, selection["gnn_cut"], ec, device)
    result = evaluate(test, predictions, metrics, output, args.method, cohort.sha256)
    record_resources(cohort, device, output, started)
    print("RUN_COMPLETE " + json.dumps(result, allow_nan=False), flush=True)


def record_resources(cohort, device, output, started):
    write_json(output / "resources.json", {
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "live_hdf5_handles_before_close": {name: len(source._files) for name, source in cohort.sources.items()
                                          if hasattr(source, "_files")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("eggnet", "exatrkx"), required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/tracking.json"))
    parser.add_argument("--metrics", required=True, help="CLAS12 train/downstream/track_finding_metrics.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
