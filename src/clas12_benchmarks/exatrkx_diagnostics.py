"""Diagnose a frozen Exa.TrkX pipeline on an explicit, unchanged event cohort.

Truth is used only for diagnostics and perfect-edge oracle labels. No model,
threshold, cohort or prediction artifact in the parent run is modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Cohort, file_sha256
from .interaction_network import SegmentClassifier
from .prepare_cohort import write_json
from .tracking import (candidates, components, filter_probabilities, graph_scores,
                       metric_module)
from .upstream import ROOT, exatrkx_embedding, exatrkx_filter, neighbor_edges


def ratio(a, b):
    return a / b if b else None


def undirected_candidates(z, k, radius=None):
    """Same proposal/union policy as production, before endpoint orientation."""
    edges = neighbor_edges(z, k, radius)
    return torch.unique(torch.sort(edges, dim=0).values, dim=1).cpu().numpy()


def graph_diagnostics(truth, edges, min_size=2):
    """Track connectivity uses only the induced true-signal edge graph.

    Paths through another track or background never establish connectivity.
    Oracle labels are components of exactly this graph, with the production
    minimum-component-size policy; disconnected pieces cannot be joined.
    """
    truth, edges = np.asarray(truth), np.asarray(edges, dtype=np.int64)
    true_edge = (truth[edges[0]] >= 0) & (truth[edges[0]] == truth[edges[1]])
    # Production treats an empty graph as all-background even at min_size=1.
    # Connectivity instead needs one distinct component per isolated node.
    cc = (components(len(truth), edges[:, true_edge], min_size=1)
          if true_edge.any() else np.arange(len(truth), dtype=np.int64))
    ids, sizes = np.unique(truth[truth >= 0], return_counts=True)
    pairs = int(np.sum(sizes * (sizes - 1) // 2))
    full = largest_hits = half = fragments = singleton_tracks = 0
    track_rows = []
    for label, size in zip(ids, sizes):
        counts = np.unique(cc[truth == label], return_counts=True)[1]
        largest = int(counts.max())
        full += len(counts) == 1
        half += 2 * largest >= size
        largest_hits += largest
        fragments += len(counts)
        singleton_tracks += size == 1
        track_rows.append({"truth_id": int(label), "hits": int(size),
                           "components": len(counts), "largest_component_hits": largest})
    # Keep production's empty-graph policy even when min_size=1. The singleton
    # labels above are only for measuring connectivity, not reconstruction.
    oracle = components(len(truth), edges[:, true_edge], min_size=min_size)
    oracle[truth < 0] = -1
    return {"true_pairs": pairs, "edges": edges.shape[1],
            "true_edges": int(true_edge.sum()), "tracks": len(ids),
            "fully_connected_tracks": int(full), "tracks_with_at_least_half_connected": int(half),
            "signal_hits": int(sizes.sum()), "largest_component_signal_hits": largest_hits,
            "true_track_components": fragments, "singleton_truth_tracks": int(singleton_tracks)}, oracle, track_rows


def aggregate_graphs(rows):
    sums = {key: sum(row[key] for row in rows) for key in rows[0]} if rows else {}
    if not rows:
        raise ValueError("No events to aggregate")
    return {**sums, "true_pair_recall": ratio(sums["true_edges"], sums["true_pairs"]),
            "edge_precision": ratio(sums["true_edges"], sums["edges"]),
            "fully_connected_track_fraction": ratio(sums["fully_connected_tracks"], sums["tracks"]),
            "at_least_half_connected_track_fraction": ratio(sums["tracks_with_at_least_half_connected"], sums["tracks"]),
            "largest_component_hit_coverage": ratio(sums["largest_component_signal_hits"], sums["signal_hits"])}


def mst_radius(distances):
    """Minimum unbounded-k radius connecting a track; zero distances are edges."""
    n = len(distances)
    if n < 2:
        return 0.
    a, b = np.triu_indices(n, 1)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    joined = 0
    for index in np.argsort(distances[a, b], kind="stable"):
        x, y = find(int(a[index])), find(int(b[index]))
        if x != y:
            parent[x] = y
            joined += 1
            if joined == n - 1:
                return float(distances[a[index], b[index]])
    raise ValueError("Complete finite distance graph was not connected")


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "quantiles": None, "cdf": None}
    probabilities = np.linspace(0, 1, 201)
    named = {str(p): float(np.quantile(values, p)) for p in (0., .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.)}
    return {"count": len(values), "mean": float(values.mean()), "quantiles": named,
            "fraction_le_1": float(np.mean(values <= 1.)),
            "fraction_le_2": float(np.mean(values <= 2.)),
            "cdf": {"distance": np.quantile(values, probabilities).tolist(),
                    "fraction": probabilities.tolist()}}


def validate_selected_stages(selected):
    names = [row["stage"] for row in selected]
    if len(names) != 3 or set(names) != {"embedding", "filter", "gnn"}:
        raise ValueError("Requires exactly one selected embedding, filter and GNN checkpoint")


def validate_frozen_sources(provenance, *, hdf5=False):
    """Verify model forwards, graph construction and the active input reader."""
    paths = ["vendor/exatrkx/exatrkx/src/embedding/layerless_embedding.py",
             "vendor/exatrkx/exatrkx/src/filter/vanilla_filter.py",
             "src/clas12_benchmarks/interaction_network.py",
             "src/clas12_benchmarks/upstream.py", "src/clas12_benchmarks/tracking.py",
             "src/clas12_benchmarks/data.py", "src/clas12_benchmarks/graphs.py",
             "src/clas12_benchmarks/sequences.py",
             "src/clas12_benchmarks/hdf5.py" if hdf5 else "src/clas12_benchmarks/sources.py"]
    saved = provenance["source_sha256"]
    for path in paths:
        if file_sha256(ROOT / path) != saved.get(path):
            raise ValueError(f"Frozen model/input source changed or missing: {path}")


def analyze(args):
    output, parent = Path(args.output), Path(args.run)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "provenance.json").exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    provenance = json.loads((parent / "provenance.json").read_text())
    if provenance["method"] != "exatrkx":
        raise ValueError("Requires an Exa.TrkX run")
    with Cohort(args.cohort) as cohort:
        return _analyze_cohort(args, output, parent, provenance, cohort)


def _analyze_cohort(args, output, parent, provenance, cohort):
    if cohort.sha256 != provenance["cohort_sha256"]:
        raise ValueError("Parent cohort fingerprint mismatch")
    if cohort.data_fingerprints() != provenance["data_array_sha256"]:
        raise ValueError("Prepared data fingerprints changed")
    validate_frozen_sources(provenance, hdf5=cohort.is_hdf5)
    metrics = metric_module(parent / "track_finding_metrics.py", output)
    if file_sha256(output / "track_finding_metrics.py") != provenance["metric_sha256"]:
        raise ValueError("Evaluator fingerprint mismatch")
    config = provenance["config"]["exatrkx"]
    selected = json.loads((parent / "selected_checkpoints.json").read_text())
    validate_selected_stages(selected)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    models = {"embedding": exatrkx_embedding(config["embedding"]),
              "filter": exatrkx_filter(config["filter"]),
              "gnn": SegmentClassifier(4, config["gnn_processing_steps"])}
    for row in selected:
        path = parent / f"{row['stage']}_best.pt"
        if file_sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint fingerprint mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["step"] != row["selected_step"]:
            raise ValueError("Selected checkpoint step mismatch")
        models[row["stage"]].load_state_dict(state["model"], strict=True)
    models = {name: model.to(device).eval() for name, model in models.items()}
    filter_cut = json.loads((parent / "filter_selection.json").read_text())["selected"]["cut"]
    gnn_cut = json.loads((parent / "frozen_selection.json").read_text())["gnn_cut"]
    count = len(cohort.roles[args.split])
    if args.max_events is not None:
        if args.max_events < 1:
            raise ValueError("max-events must be positive")
        count = min(count, args.max_events)
    if not count:
        raise ValueError("No events selected")
    write_json(output / "provenance.json", {"parent_run": str(parent), "cohort_sha256": cohort.sha256,
        "split": args.split, "events": count, "selection": "first events in frozen cohort order; all by default",
        "device": str(device), "torch": torch.__version__, "selected_checkpoints": selected,
        "edge_orientation": config.get("edge_orientation", "legacy_index"), "config": config,
        "filter_cut": filter_cut, "gnn_cut": gnn_cut, "data_fingerprints_verified": True,
        "metric_sha256": provenance["metric_sha256"],
        "diagnostic_source_sha256": file_sha256(Path(__file__)),
        "oracle_policy": "remove every false/background edge; connected components; reject components smaller than the saved minimum; never join disconnected components",
        "connectivity_policy": "fully connected induced true-track subgraph; no paths via other tracks/noise; singleton tracks connected trivially",
        "candidate_policy": "union of directed k-nearest proposals with Euclidean radius <= cutoff, deduplicated as undirected pairs",
        "distance_weighting": "unordered pairs pooled over events; track span and MST radii pooled with one entry per true track"})
    stages = ("candidate", "after_filter", "after_gnn")
    graphs = {stage: [] for stage in stages}
    oracle_rows = {stage: [] for stage in stages}
    actual_rows = []
    radius_rows = [[] for _ in args.radii]
    k_rows = [[] for _ in args.ks]
    distances = {key: [] for key in ("same_track", "different_signal_tracks", "background_involved")}
    track_span, track_mst = [], []
    losses = {"fully_connected_tracks_lost_at_filter": 0, "fully_connected_tracks_lost_at_gnn": 0}
    with (output / "per_event.jsonl").open("w") as events_file, (output / "per_track.jsonl").open("w") as tracks_file, torch.no_grad():
        for index in range(count):
            event = cohort.event(args.split, index)
            truth = event.labels["seg_target"]
            z = models["embedding"](torch.as_tensor(event.positions, device=device))
            if not torch.isfinite(z).all():
                raise ValueError("Nonfinite embedding")
            distance = torch.cdist(z, z, compute_mode="donot_use_mm_for_euclid_dist").cpu().numpy()
            a, b = np.triu_indices(len(truth), 1)
            same = (truth[a] >= 0) & (truth[a] == truth[b])
            different = (truth[a] >= 0) & (truth[b] >= 0) & ~same
            distances["same_track"].append(distance[a[same], b[same]])
            distances["different_signal_tracks"].append(distance[a[different], b[different]])
            bg = (truth[a] < 0) | (truth[b] < 0)
            distances["background_involved"].append(distance[a[bg], b[bg]])
            for label in np.unique(truth[truth >= 0]):
                nodes = np.flatnonzero(truth == label)
                sub = distance[np.ix_(nodes, nodes)]
                track_span.append(float(sub.max()))
                track_mst.append(mst_radius(sub))
            unlimited_radius = undirected_candidates(z, config["candidate_k"])
            for i, radius in enumerate(args.radii):
                edge = unlimited_radius[:, distance[tuple(unlimited_radius)] <= radius]
                row, _, _ = graph_diagnostics(truth, edge, config["min_component_size"])
                radius_rows[i].append(row)
            for i, k in enumerate(args.ks):
                edge = undirected_candidates(z, max(1, len(truth) - 1) if k is None else k, config["candidate_radius"])
                row, _, _ = graph_diagnostics(truth, edge, config["min_component_size"])
                k_rows[i].append(row)
            candidate = candidates(models["embedding"], event, config, device)
            filtered = candidate[:, filter_probabilities(models["filter"], event, candidate, device) >= filter_cut]
            scores = graph_scores(models["gnn"], [event], [filtered], device)[0]
            classified = filtered[:, scores >= gnn_cut]
            event_rows, track_rows = {}, {}
            for stage, edge in zip(stages, (candidate, filtered, classified)):
                row, oracle, per_track = graph_diagnostics(truth, edge.numpy(), config["min_component_size"])
                graphs[stage].append(row)
                oracle_metrics = metrics.event_track_metrics(truth, oracle)
                oracle_rows[stage].append(oracle_metrics)
                event_rows[stage] = {"graph": row, "oracle": oracle_metrics}
                track_rows[stage] = per_track
            for i, radius in enumerate(args.radii):
                if radius == config["candidate_radius"] and radius_rows[i][-1] != graphs["candidate"][-1]:
                    raise ValueError("Radius-scan baseline differs from production candidates")
            for i, k in enumerate(args.ks):
                if k == config["candidate_k"] and k_rows[i][-1] != graphs["candidate"][-1]:
                    raise ValueError("k-scan baseline differs from production candidates")
            for i in range(len(track_rows["candidate"])):
                c, f, g = (track_rows[stage][i] for stage in stages)
                losses["fully_connected_tracks_lost_at_filter"] += c["components"] == 1 and f["components"] != 1
                losses["fully_connected_tracks_lost_at_gnn"] += f["components"] == 1 and g["components"] != 1
                tracks_file.write(json.dumps({"event": event.identity, "truth_id": c["truth_id"], "hits": c["hits"],
                    "by_stage": {stage: track_rows[stage][i] for stage in stages}}, allow_nan=False) + "\n")
            actual = components(len(truth), classified.numpy(), config["min_component_size"])
            row = metrics.event_track_metrics(truth, actual)
            actual_rows.append(row)
            events_file.write(json.dumps({"identity": event.identity, "stages": event_rows, "actual_gnn": row}, allow_nan=False) + "\n")
            if (index + 1) % 250 == 0:
                print(json.dumps({"events_completed": index + 1, "events_total": count}), flush=True)
    radius_scan = [{"radius": None if np.isinf(radius) else radius, "k": config["candidate_k"], **aggregate_graphs(rows)}
                   for radius, rows in zip(args.radii, radius_rows)]
    k_scan = [{"k": k, "radius": config["candidate_radius"], **aggregate_graphs(rows)} for k, rows in zip(args.ks, k_rows)]
    stage_summary = {stage: {"graph": aggregate_graphs(graphs[stage]),
                            "oracle": metrics.summarize_event_metrics(oracle_rows[stage])} for stage in stages}
    summary = {"events": count, "split": args.split, "cohort_sha256": cohort.sha256,
               "stages": stage_summary, "actual_gnn": metrics.summarize_event_metrics(actual_rows),
               "connectivity_losses": {k: int(v) for k, v in losses.items()},
               "filter_true_edge_recall_given_candidates": ratio(stage_summary["after_filter"]["graph"]["true_edges"], stage_summary["candidate"]["graph"]["true_edges"]),
               "gnn_true_edge_recall_given_filter": ratio(stage_summary["after_gnn"]["graph"]["true_edges"], stage_summary["after_filter"]["graph"]["true_edges"])}
    if args.split == "val" and count == len(cohort.roles["val"]):
        frozen = json.loads((parent / "frozen_selection.json").read_text())
        keys = ("ari_signal", "track_purity_global", "track_efficiency_global")
        summary["replay_minus_saved_validation"] = {
            k: summary["actual_gnn"][k] - frozen["metrics"][k]
            if summary["actual_gnn"][k] is not None and frozen["metrics"][k] is not None else None
            for k in keys}
        recorded = json.loads((parent / "candidate_diagnostics.json").read_text())["val"]
        summary["recorded_candidate_diagnostics"] = recorded
        if stage_summary["candidate"]["graph"]["true_pairs"] != recorded["true_pairs"]:
            raise ValueError("True-pair population changed")
    dist = {key: distribution(np.concatenate(parts)) for key, parts in distances.items()}
    dist.update(track_maximum_pair_distance=distribution(track_span), track_mst_connection_radius=distribution(track_mst))
    for name, content in (("summary.json", summary), ("radius_scan.json", radius_scan), ("k_scan.json", k_scan),
                          ("distance_distributions.json", dist)):
        write_json(output / name, content)
    print("EXATRKX_DIAGNOSTICS_COMPLETE " + json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--radii", nargs="+", type=float, default=[0., .1, .2, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., float("inf")])
    parser.add_argument("--ks", nargs="+", default=["1", "2", "4", "8", "16", "24", "32", "48", "64", "96", "128", "all"])
    args = parser.parse_args()
    args.ks = [None if str(k).lower() == "all" else int(k) for k in args.ks]
    if any(k is not None and k < 1 for k in args.ks) or any(np.isnan(r) or r < 0 for r in args.radii):
        parser.error("k must be positive or all; radii must be nonnegative or inf")
    analyze(args)


if __name__ == "__main__":
    main()
