"""Sweep only the GNN probability cut of a frozen Exa.TrkX validation run."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Cohort, file_sha256
from .exatrkx_diagnostics import (aggregate_graphs, graph_diagnostics, ratio,
                                  validate_frozen_sources, validate_selected_stages)
from .interaction_network import SegmentClassifier
from .prepare_cohort import write_json
from .tracking import candidates, components, filter_probabilities, graph_scores, metric_module
from .upstream import exatrkx_embedding, exatrkx_filter
from .finding_evaluation import FindingEvaluation, finding_fingerprints, finding_sequence
from .finding_statistics import mc_intervals


def strict_track_states(truth, prediction, track_rows):
    """A clean connected track shares its component with no other signal track.

    Background contamination is allowed here and measured separately. True
    connectivity was already computed using only within-track true edges.
    """
    component_truth = {int(p): set(truth[(prediction == p) & (truth >= 0)].tolist())
                       for p in np.unique(prediction[prediction >= 0])}
    full, clean, exposed = [], [], []
    for row in track_rows:
        label = row["truth_id"]
        ids = np.unique(prediction[truth == label])
        connected = row["components"] == 1
        full.append(connected)
        clean.append(connected and len(ids) == 1 and ids[0] >= 0 and component_truth[int(ids[0])] == {label})
        exposed.append(any(p >= 0 and len(component_truth[int(p)]) > 1 for p in ids))
    return np.asarray(full, dtype=bool), np.asarray(clean, dtype=bool), np.asarray(exposed, dtype=bool), sum(len(v) > 1 for v in component_truth.values())


def recovery_counts(baseline, candidate):
    base_full, base_clean, _, _ = baseline
    full, clean, exposed, _ = candidate
    recovered = ~base_full & full
    return {"newly_fully_connected_tracks": int(recovered.sum()),
            "newly_fully_connected_unmerged_tracks": int((recovered & clean).sum()),
            "newly_fully_connected_merged_tracks": int((recovered & exposed).sum()),
            "baseline_clean_tracks_now_merge_exposed": int((base_clean & exposed).sum()),
            "baseline_clean_tracks_no_longer_clean": int((base_clean & ~clean).sum())}


def paired_delta_bootstrap(candidate, baseline, seed=1337, replicates=1000):
    """Event-paired descriptive intervals, conditional on frozen trained weights."""
    def values(rows):
        return np.array([[r["metrics"]["n_matched_tracks"], r["metrics"]["n_true_tracks"],
                          r["metrics"]["n_pred_tracks"], np.nan if r["metrics"]["ari_signal"] is None else r["metrics"]["ari_signal"],
                          r["graph"]["fully_connected_tracks"], r["clean_fully_connected_tracks"]] for r in rows], dtype=float)
    a, b = values(candidate), values(baseline)
    if a.shape != b.shape or not len(a):
        raise ValueError("Paired bootstrap needs the same nonempty event population")
    def divide(x, y):
        return np.divide(x, y, out=np.full_like(x, np.nan), where=y != 0)
    def summarize(draws):
        sums = np.nansum(draws, axis=1)
        ari = divide(np.nansum(draws[:, :, 3], axis=1), np.isfinite(draws[:, :, 3]).sum(axis=1))
        return np.column_stack((divide(2*sums[:, 0], sums[:, 1]+sums[:, 2]), ari,
                                divide(sums[:, 0], sums[:, 2]), divide(sums[:, 0], sums[:, 1]),
                                divide(sums[:, 4], sums[:, 1]), divide(sums[:, 5], sums[:, 1])))
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, replicates, 32):
        indices = rng.integers(len(a), size=(min(32, replicates-start), len(a)))
        draws.extend((summarize(a[indices])-summarize(b[indices])).tolist())
    draws = np.asarray(draws)
    keys = ("track_f1_global", "ari_signal", "track_purity_global", "track_efficiency_global",
            "fully_connected_track_fraction", "clean_fully_connected_track_fraction")
    intervals = {}
    for i, key in enumerate(keys):
        finite = draws[:, i][np.isfinite(draws[:, i])]
        intervals[key] = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None
    if "mc_track_dice_sum" in candidate[0]["metrics"]:
        extra = mc_intervals([r["metrics"] for r in candidate], [r["metrics"] for r in baseline],
                             seed=seed, replicates=replicates)
        intervals.update(extra["intervals_95"])
    return {"unit": "paired event", "seed": seed, "replicates": replicates,
            "delta_95_intervals": intervals,
            "scope": "descriptive validation intervals; conditional on weights; threshold selection and evaluation use the same validation set"}


def sweep(args):
    parent, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "provenance.json").exists():
        raise FileExistsError("Use a fresh sweep output directory")
    old = json.loads((parent / "provenance.json").read_text())
    if old["method"] != "exatrkx":
        raise ValueError("Requires an Exa.TrkX run")
    with Cohort(args.cohort) as cohort:
        return _sweep_cohort(args, parent, output, old, cohort)


def validation_report(curve, rows, saved_cut, *, mc=False, replicates=1000):
    """Descriptive selections and paired intervals; never change the saved cut."""
    baseline = next(row for row in curve if row["cut"] == saved_cut)
    best = max(curve, key=lambda r: r["track_f1_global"] if r["track_f1_global"] is not None else -1.)
    lower = [row for row in curve if row["cut"] < saved_cut]
    best_lower = max(lower, key=lambda r: r["track_f1_global"] if r["track_f1_global"] is not None else -1.) if lower else None
    best_ari = max(curve, key=lambda r: r["metrics"]["ari_signal"] if r["metrics"]["ari_signal"] is not None else -1.)
    best_clean = max(curve, key=lambda r: r["clean_fully_connected_track_fraction"] or -1.)
    selections = {"baseline": baseline, "best_by_f1": best, "best_lower_cut_by_f1": best_lower,
                  "best_by_ari": best_ari, "best_by_clean_connectivity": best_clean}
    if mc:
        selections["best_by_mc_dice"] = max(curve, key=lambda r: r["metrics"]["mc_track_dice_mean"]
                                            if r["metrics"]["mc_track_dice_mean"] is not None else -1.)
    comparisons = {}
    for row in selections.values():
        if row is not None and row["cut"] != saved_cut and str(row["cut"]) not in comparisons:
            comparisons[str(row["cut"])] = paired_delta_bootstrap(rows[row["cut"]], rows[saved_cut], replicates=replicates)
    return {"saved_cut": saved_cut, **selections, "paired_delta_vs_saved_cut": comparisons, "validation_only": True}


def _sweep_cohort(args, parent, output, old, cohort):
    if cohort.sha256 != old["cohort_sha256"]:
        raise ValueError("Parent cohort fingerprint mismatch")
    if cohort.data_fingerprints() != old["data_array_sha256"]:
        raise ValueError("Prepared data fingerprints changed")
    validate_frozen_sources(old, hdf5=cohort.is_hdf5)
    metrics = metric_module(parent / "track_finding_metrics.py", output)
    if file_sha256(output / "track_finding_metrics.py") != old["metric_sha256"]:
        raise ValueError("Evaluator fingerprint mismatch")
    evaluation = FindingEvaluation(metrics, old.get("finding_evaluation"))
    if "finding_evaluation" in old and finding_fingerprints(cohort, evaluation) != old.get("finding_data_sha256"):
        raise ValueError("Finding provenance sidecars changed")
    config = old["config"]["exatrkx"]
    selected = json.loads((parent / "selected_checkpoints.json").read_text())
    validate_selected_stages(selected)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    models = {"embedding": exatrkx_embedding(config["embedding"]), "filter": exatrkx_filter(config["filter"]),
              "gnn": SegmentClassifier(4, config["gnn_processing_steps"])}
    for row in selected:
        path = parent / f"{row['stage']}_best.pt"
        if file_sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint fingerprint mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["step"] != row["selected_step"]:
            raise ValueError("Selected checkpoint step mismatch")
        models[row["stage"]].load_state_dict(state["model"], strict=True)
    models = {k: v.to(device).eval() for k, v in models.items()}
    filter_cut = json.loads((parent / "filter_selection.json").read_text())["selected"]["cut"]
    saved_cut = float(json.loads((parent / "frozen_selection.json").read_text())["gnn_cut"])
    cuts = sorted(set([*args.cuts, saved_cut]))
    count = len(cohort.roles["val"])
    if args.max_events is not None:
        if args.max_events < 1:
            raise ValueError("max-events must be positive")
        count = min(count, args.max_events)
    if not count:
        raise ValueError("No validation events")
    events = finding_sequence(cohort, "val", evaluation, count)
    write_json(output / "provenance.json", {"parent_run": str(parent), "cohort_sha256": cohort.sha256, "events": count,
        "split": "val", "device": str(device), "torch": torch.__version__, "selected_checkpoints": selected,
        "config": config, "filter_cut": filter_cut, "saved_gnn_cut": saved_cut, "cuts": cuts,
        "data_fingerprints_verified": True, "metric_sha256": old["metric_sha256"], "source_sha256": file_sha256(Path(__file__)),
        "only_variable": "GNN probability cut; scores computed once per event; edge retained when score >= cut",
        "f1_definition": "2*pooled matched tracks / (pooled truth tracks + pooled predicted tracks)",
        "connectivity_definition": "all hits of each true track connected by retained true edges only",
        "clean_connectivity_definition": "fully true-connected track with no other signal-track hit in its reconstructed component; background contamination allowed and reported separately",
        "merge_rate_definition": "frozen evaluator mean over events of fraction of predicted tracks containing hits from multiple true signal tracks",
        "no_test_evaluation": True, "parent_run_unchanged": True,
        **(evaluation.metadata() if "finding_evaluation" in old else {})})
    rows = {cut: [] for cut in cuts}
    cache_edges, cache_scores, offsets, point_counts = [], [], [0], []
    with torch.no_grad(), (output / "per_event.jsonl").open("w") as stream:
        for index in range(count):
            event = events[index]
            truth = event.labels["seg_target"]
            candidate = candidates(models["embedding"], event, config, device)
            filtered = candidate[:, filter_probabilities(models["filter"], event, candidate, device) >= filter_cut]
            score = graph_scores(models["gnn"], [event], [filtered], device)[0].numpy()
            if not np.isfinite(score).all():
                raise ValueError("Nonfinite GNN score")
            filtered = filtered.numpy()
            cache_edges.append(filtered); cache_scores.append(score)
            offsets.append(offsets[-1] + len(score)); point_counts.append(len(truth))
            states, records = {}, {}
            for cut in cuts:
                edges = filtered[:, score >= cut]
                graph, _, tracks = graph_diagnostics(truth, edges, config["min_component_size"])
                prediction = components(len(truth), edges, config["min_component_size"])
                full, clean, exposed, merged = strict_track_states(truth, prediction, tracks)
                states[cut] = full, clean, exposed, merged
                metric_row = evaluation.event_metrics(event, prediction)
                if "finding_evaluation" in old:
                    metric_row.update(identity=event.identity, finding_evaluation=evaluation.contract)
                row = {"graph": graph, "metrics": metric_row,
                       "clean_fully_connected_tracks": int(clean.sum()), "merge_exposed_true_tracks": int(exposed.sum()),
                       "merged_predicted_tracks": int(merged)}
                records[cut] = row
            for cut in cuts:
                records[cut]["recovery_vs_saved_cut"] = recovery_counts(states[saved_cut], states[cut])
                rows[cut].append(records[cut])
            stream.write(json.dumps({"identity": event.identity, "by_cut": records}, allow_nan=False) + "\n")
            if (index+1) % 250 == 0:
                print(json.dumps({"events_completed": index+1, "events_total": count}), flush=True)
    np.savez_compressed(output / "edge_scores.npz", edges=np.concatenate(cache_edges, axis=1),
                        scores=np.concatenate(cache_scores), event_offsets=np.asarray(offsets),
                        event_point_counts=np.asarray(point_counts))
    curve = []
    for cut in cuts:
        records = rows[cut]
        m = evaluation.summarize([r["metrics"] for r in records])
        g = aggregate_graphs([r["graph"] for r in records])
        clean = sum(r["clean_fully_connected_tracks"] for r in records)
        recovery = {key: sum(r["recovery_vs_saved_cut"][key] for r in records) for key in records[0]["recovery_vs_saved_cut"]}
        curve.append({"cut": cut, "track_f1_global": ratio(2*m["n_matched_tracks"], m["n_true_tracks"]+m["n_pred_tracks"]),
                      "metrics": m, "graph": g, "clean_fully_connected_tracks": clean,
                      "clean_fully_connected_track_fraction": ratio(clean, g["tracks"]),
                      "merge_exposed_true_track_fraction": ratio(sum(r["merge_exposed_true_tracks"] for r in records), g["tracks"]),
                      "merged_predicted_track_fraction_global": ratio(sum(r["merged_predicted_tracks"] for r in records), m["n_pred_tracks"]),
                      "recovery_vs_saved_cut": recovery})
    report = {"events": count, "split": "val", "cohort_sha256": cohort.sha256,
              **(evaluation.metadata() if "finding_evaluation" in old else {}),
              **validation_report(curve, rows, saved_cut, mc=evaluation.mc, replicates=args.bootstrap)}
    for row in selected:
        if file_sha256(parent / f"{row['stage']}_best.pt") != row["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint changed during analysis")
    write_json(output / "threshold_curve.json", curve)
    write_json(output / "summary.json", report)
    print("GNN_THRESHOLD_SWEEP_COMPLETE " + json.dumps({"saved_cut": saved_cut, "best_f1_cut": report["best_by_f1"]["cut"],
          "best_ari_cut": report["best_by_ari"]["cut"], "best_clean_connectivity_cut": report["best_by_clean_connectivity"]["cut"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--cuts", nargs="+", type=float, default=[0., .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.])
    args = parser.parse_args()
    if args.bootstrap < 1 or any(not np.isfinite(cut) or cut < 0 or cut > 1 for cut in args.cuts):
        parser.error("Cuts must be finite probabilities in [0,1]; bootstrap must be positive")
    sweep(args)


if __name__ == "__main__":
    main()
