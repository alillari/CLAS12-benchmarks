"""Run one preselected recipe and compare frozen v1 on identical fresh events."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from clas12_benchmarks.prepare_cohort import write_json
from clas12_benchmarks.data import Cohort


def paired_bootstrap(tuned_path, frozen_path, seed=1337, replicates=1000):
    def read(path):
        with path.open() as stream:
            return [json.loads(line) for line in stream]
    a, b = read(tuned_path), read(frozen_path)
    if len(a) != len(b) or any(x["identity"] != y["identity"] for x, y in zip(a, b)):
        raise ValueError("Paired evaluation identities differ")
    if not a or replicates < 1:
        raise ValueError("Paired bootstrap requires events and positive replicates")
    columns = ("n_matched_tracks", "n_true_tracks", "n_pred_tracks", "ari_signal")
    x = np.array([[np.nan if row[key] is None else row[key] for key in columns] for row in a])
    y = np.array([[np.nan if row[key] is None else row[key] for key in columns] for row in b])
    rng = np.random.default_rng(seed)
    draws = []
    def ratio(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)
    def event_mean(values):
        return ratio(np.nansum(values, axis=1), np.isfinite(values).sum(axis=1))
    for begin in range(0, replicates, 32):
        index = rng.integers(len(a), size=(min(32, replicates - begin), len(a)))
        xs, ys = x[index], y[index]
        xv, yv = xs[:, :, :3].sum(1), ys[:, :, :3].sum(1)
        draws.extend(np.column_stack((ratio(xv[:, 0], xv[:, 1]) - ratio(yv[:, 0], yv[:, 1]),
            ratio(xv[:, 0], xv[:, 2]) - ratio(yv[:, 0], yv[:, 2]),
            event_mean(xs[:, :, 3]) - event_mean(ys[:, :, 3]))).tolist())
    values = np.asarray(draws)
    intervals, defined = {}, {}
    for i, name in enumerate(("track_efficiency_global", "track_purity_global", "ari_signal")):
        finite = values[:, i][np.isfinite(values[:, i])]
        intervals[name] = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None
        defined[name] = len(finite)
    return {"unit": "event; same resampled indices for both methods", "replicates": replicates, "seed": seed,
        "delta_95_percent_intervals": intervals, "defined_resamples": defined}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("eggnet", "exatrkx"), required=True)
    parser.add_argument("--cohort", default="/cohort/cohort.json")
    parser.add_argument("--original-cohort", default="/previous/cohort.json")
    parser.add_argument("--parent", default="/original")
    parser.add_argument("--metrics", default="/clas12-metrics/track_finding_metrics.py")
    parser.add_argument("--config", default="/workspace/configs/mild_tuning_20261103.json")
    parser.add_argument("--output", default="/results")
    args = parser.parse_args()
    output = Path(args.output)
    tuned, frozen = output / "tuned", output / "original_frozen"
    def run(argv):
        subprocess.run([sys.executable, "-u", *argv], check=True)
    run(["-m", "clas12_benchmarks.tracking", "--method", args.method, "--cohort", args.cohort,
         "--config", args.config, "--metrics", args.metrics, "--output", str(tuned)])
    run(["-m", "clas12_benchmarks.evaluate_frozen", "--parent", args.parent,
         "--original-cohort", args.original_cohort, "--cohort", args.cohort, "--output", str(frozen)])
    for name, root in (("tuned", tuned), ("original_frozen", frozen)):
        run([str(Path(__file__).with_name("audit_tracking_predictions.py")), "--run", str(root),
             "--cohort", args.cohort, "--output", str(output / f"{name}_audit.json")])
    spec = importlib.util.spec_from_file_location("comparison_frozen_metrics", tuned / "track_finding_metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    a, b = (json.loads((root / "summary.json").read_text()) for root in (tuned, frozen))
    if a["coatjava"] != b["coatjava"]:
        raise ValueError("COATJAVA populations/metrics differ")
    cohort = Cohort(args.cohort)
    report = {"method": args.method, "cohort_sha256": cohort.sha256,
        "tuned": a, "original_frozen": b,
        "tuned_minus_original": metrics.compare_metric_summaries(a["metrics"], b["metrics"]),
        "selected_checkpoints": json.loads((tuned / "selected_checkpoints.json").read_text()),
        "selection": json.loads((tuned / "frozen_selection.json").read_text()),
        "paired_bootstrap": paired_bootstrap(tuned / "per_event_metrics.jsonl", frozen / "per_event_metrics.jsonl"),
        "audits_passed": True}
    write_json(output / "comparison.json", report)
    print("COMPARISON_COMPLETE " + json.dumps(report["tuned_minus_original"]), flush=True)


if __name__ == "__main__":
    main()
