"""Run the fixed tracking recipe across nested training-event label budgets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.prepare_cohort import write_json


def bootstrap_metrics(path, seed=1337, replicates=1000):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    columns = ("n_matched_tracks", "n_true_tracks", "n_pred_tracks", "ari_signal", "background_rejection")
    values = np.array([[np.nan if r[k] is None else r[k] for k in columns] for r in rows])
    rng, samples = np.random.default_rng(seed), []
    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    def event_mean(a):
        return ratio(np.nansum(a, axis=1), np.isfinite(a).sum(axis=1))
    for start in range(0, replicates, 32):
        draws = values[rng.integers(len(rows), size=(min(32, replicates - start), len(rows)))]
        sums = draws[:, :, :3].sum(1)
        samples.extend(np.column_stack((ratio(sums[:, 0], sums[:, 1]), ratio(sums[:, 0], sums[:, 2]),
            ratio(2 * sums[:, 0], sums[:, 1] + sums[:, 2]), event_mean(draws[:, :, 3]),
            event_mean(draws[:, :, 4]))).tolist())
    samples = np.asarray(samples)
    keys = ("track_efficiency_global", "track_purity_global", "track_f1_global", "ari_signal", "background_rejection")
    intervals, defined = {}, {}
    for i, key in enumerate(keys):
        finite = samples[:, i][np.isfinite(samples[:, i])]
        intervals[key] = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None
        defined[key] = len(finite)
    return {"unit": "event", "replicates": replicates, "seed": seed,
        "intervals_95": intervals, "defined_resamples": defined}


def verify_manifests(cohorts, budgets):
    previous = set()
    heldout = None
    for budget in budgets:
        spec = json.loads((Path(cohorts) / f"cohort_{budget}.json").read_text())
        members = {(m["dataset"], m["index"]) for m in spec["roles"]["train"]}
        if len(members) != budget or not previous <= members:
            raise ValueError("Training budgets are not nested/exact")
        current = {role: spec["roles"][role] for role in ("val", "test")}
        if heldout is not None and current != heldout:
            raise ValueError("Holdout membership/order changed")
        heldout, previous = current, members


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("eggnet", "exatrkx"), required=True)
    p.add_argument("--cohorts", default="/cohorts")
    p.add_argument("--config", default="/workspace/configs/label_budget_sweep_20261103.json")
    p.add_argument("--metrics", default="/clas12-metrics/track_finding_metrics.py")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="/results")
    p.add_argument("--budgets", nargs="+", type=int, default=[100, 1000, 10000, 100000])
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "sweep_progress.json").exists():
        raise FileExistsError("Existing sweep output; retain it and use a fresh run ID")
    verify_manifests(args.cohorts, args.budgets)
    report = {"method": args.method, "source_commit": os.environ.get("BENCHMARK_GIT_COMMIT"),
        "config_sha256": file_sha256(args.config), "label_budget_unit": "fully labeled training events",
        "validation_labels_excluded_from_training_budget": True, "arms": {}}
    baseline = None
    for budget in args.budgets:
        arm = output / f"labels_{budget}"
        arm.mkdir()
        manifest = Path(args.cohorts) / f"cohort_{budget}.json"
        progress = {"method": args.method, "active_budget": budget, "phase": "training_and_evaluation",
                    "completed_budgets": list(report["arms"])}
        write_json(output / "sweep_progress.json", progress)
        print(f"LABEL_BUDGET_START {args.method} {budget}", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "clas12_benchmarks.tracking", "--method", args.method,
            "--config", args.config, "--cohort", str(manifest), "--metrics", args.metrics,
            "--output", str(arm / "run"), "--device", args.device], check=True)
        progress["phase"] = "prediction_audit"
        write_json(output / "sweep_progress.json", progress)
        subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("audit_tracking_predictions.py")),
            "--run", str(arm / "run"), "--cohort", str(manifest), "--output", str(arm / "audit.json")], check=True)
        summary = json.loads((arm / "run/summary.json").read_text())
        if baseline is not None and summary["coatjava"] != baseline:
            raise ValueError("Evaluation population/COATJAVA changed across label budgets")
        baseline = summary["coatjava"]
        selected = json.loads((arm / "run/selected_checkpoints.json").read_text())
        for row in selected:
            row["distinct_training_events_seen_by_selected_checkpoint"] = min(budget, row["selected_step"])
        metrics = summary["metrics"]
        report["arms"][str(budget)] = {"training_events": budget,
            "cohort_sha256": file_sha256(manifest), "summary": summary,
            "track_f1_global": 2 * metrics["n_matched_tracks"] / (metrics["n_true_tracks"] + metrics["n_pred_tracks"]),
            "selected_checkpoints": selected,
            "selection": json.loads((arm / "run/frozen_selection.json").read_text()),
            "bootstrap": bootstrap_metrics(arm / "run/per_event_metrics.jsonl"),
            "audit": json.loads((arm / "audit.json").read_text())}
        write_json(output / "sweep_partial.json", report)
        print("LABEL_BUDGET_COMPLETE " + json.dumps({"method": args.method, "budget": budget,
            "track_f1_global": report["arms"][str(budget)]["track_f1_global"], **metrics}), flush=True)
    write_json(output / "sweep_summary.json", report)
    write_json(output / "sweep_progress.json", {"method": args.method, "phase": "complete", "completed_budgets": list(report["arms"])})
    print("LABEL_SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
