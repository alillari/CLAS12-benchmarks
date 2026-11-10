"""Run one preselected recipe and compare frozen v1 on identical fresh events."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from clas12_benchmarks.prepare_cohort import write_json
from clas12_benchmarks.data import Cohort
from clas12_benchmarks.finding_evaluation import DEFAULT_METRICS, FindingEvaluation, load_metrics, require_same_contract, require_same_population
from clas12_benchmarks.finding_statistics import mc_intervals
from clas12_benchmarks.rescore_finding import rescore


def paired_bootstrap(tuned_path, frozen_path, seed=1337, replicates=1000):
    def read(path):
        with path.open() as stream:
            return [json.loads(line) for line in stream]
    a, b = read(tuned_path), read(frozen_path)
    if len(a) != len(b) or any(x["identity"] != y["identity"] for x, y in zip(a, b)):
        raise ValueError("Paired evaluation identities differ")
    if not a or replicates < 1:
        raise ValueError("Paired bootstrap requires events and positive replicates")
    for x, y in zip(a, b, strict=True):
        require_same_contract(a[0], x)
        require_same_contract(a[0], y)
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
    if "mc_track_dice_sum" in a[0]:
        extra = mc_intervals(a, b, seed=seed, replicates=replicates)
        intervals.update(extra["intervals_95"])
        defined.update(extra["defined_resamples"])
    return {"unit": "event; same resampled indices for both methods", "replicates": replicates, "seed": seed,
        "delta_95_percent_intervals": intervals, "defined_resamples": defined}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("eggnet", "exatrkx"), required=True)
    parser.add_argument("--cohort", default="/cohort/cohort.json")
    parser.add_argument("--original-cohort", default="/previous/cohort.json")
    parser.add_argument("--parent", default="/original")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/tracking.json"))
    parser.add_argument("--output", default="/results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    tuned, frozen = output / "tuned", output / "original_frozen"
    def run(argv):
        subprocess.run([sys.executable, "-u", *argv], check=True)
    run(["-m", "clas12_benchmarks.tracking", "--method", args.method, "--cohort", args.cohort,
         "--config", args.config, "--metrics", args.metrics, "--output", str(tuned),
         "--device", args.device, "--threads", str(args.threads)])
    run(["-m", "clas12_benchmarks.evaluate_frozen", "--parent", args.parent,
         "--original-cohort", args.original_cohort, "--cohort", args.cohort, "--output", str(frozen),
         "--device", args.device, "--threads", str(args.threads)])
    for name, root in (("tuned", tuned), ("original_frozen", frozen)):
        run([str(Path(__file__).with_name("audit_tracking_predictions.py")), "--run", str(root),
             "--cohort", args.cohort, "--output", str(output / f"{name}_audit.json")])
    metrics = load_metrics(tuned / "track_finding_metrics.py")
    a, b = (json.loads((root / "summary.json").read_text()) for root in (tuned, frozen))
    historical = b
    comparison_frozen = frozen
    if "finding_evaluation" in a:
        comparison_frozen = output / "original_rescored"
        b = rescore(frozen, args.cohort, comparison_frozen, metrics_path=tuned / "track_finding_metrics.py",
                    settings=a["finding_evaluation"])
    require_same_population(a, b)
    if a["coatjava"] != b["coatjava"]:
        raise ValueError("COATJAVA populations/metrics differ")
    cohort = Cohort(args.cohort)
    report = {"method": args.method, "cohort_sha256": cohort.sha256,
        "tuned": a, "original_frozen": b, "historical_original_frozen": historical,
        "tuned_minus_original": FindingEvaluation(metrics, a.get("finding_evaluation")).compare(a["metrics"], b["metrics"]),
        "selected_checkpoints": json.loads((tuned / "selected_checkpoints.json").read_text()),
        "selection": json.loads((tuned / "frozen_selection.json").read_text()),
        "paired_bootstrap": paired_bootstrap(tuned / "per_event_metrics.jsonl", comparison_frozen / "per_event_metrics.jsonl"),
        "frozen_comparison_evaluation": str(comparison_frozen),
        "audits_passed": True}
    write_json(output / "comparison.json", report)
    print("COMPARISON_COMPLETE " + json.dumps(report["tuned_minus_original"]), flush=True)


if __name__ == "__main__":
    main()
