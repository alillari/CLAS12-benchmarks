"""Recompute shared metrics from every saved point prediction after a run."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.prepare_cohort import write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    cohort = Cohort(args.cohort)
    provenance = json.loads((root / "provenance.json").read_text())
    require(provenance["cohort_sha256"] == cohort.sha256, "Cohort fingerprint mismatch")
    metric_path = root / "track_finding_metrics.py"
    require(file_sha256(metric_path) == provenance["metric_sha256"], "Evaluator fingerprint mismatch")
    spec = importlib.util.spec_from_file_location("frozen_tracking_metric_audit", metric_path)
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    paths = sorted((root / "predictions").glob("*.npz"))
    require(len(paths) == len(cohort.roles["test"]), "Prediction event count mismatch")
    with (root / "per_event_metrics.jsonl").open() as stream:
        cached_rows = [json.loads(line) for line in stream]
    require(len(cached_rows) == len(paths), "Per-event metric count mismatch")
    rows, baseline = [], []
    for i, path in enumerate(paths):
        require(path.name == f"event_{i:06d}.npz", "Prediction filename/order mismatch")
        event = cohort.event("test", i)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            require(metadata["schema"] == "clas12_baseline_predictions_v1", "Prediction schema mismatch")
            require(metadata["cohort_sha256"] == cohort.sha256, "Prediction cohort mismatch")
            require(metadata["identity"] == event.identity, "Prediction event identity mismatch")
            require(metadata["method"] == provenance["method"], "Prediction method mismatch")
            prediction = saved["predictions"]
            require(prediction.shape == (len(event.positions),), "Prediction point count mismatch")
            require(np.issubdtype(prediction.dtype, np.integer), "Prediction labels must be integers")
            np.testing.assert_array_equal(saved["original_point_index"], event.original_point_index)
            row = metrics.event_track_metrics(event.labels["seg_target"], prediction)
            require(cached_rows[i] == {"identity": event.identity, **row},
                    f"Per-event metrics or identity differ from predictions: event {i}")
            rows.append(row)
            baseline.append(metrics.event_track_metrics(event.labels["seg_target"], event.labels["coatjava_seg_pred"]))
    actual = {"metrics": metrics.summarize_event_metrics(rows), "coatjava": metrics.summarize_event_metrics(baseline)}
    expected = json.loads((root / "summary.json").read_text())
    for group, values in actual.items():
        for key, value in values.items():
            if value is None:
                require(expected[group][key] is None, f"Undefined summary metric mismatch: {group}/{key}")
            else:
                np.testing.assert_allclose(value, expected[group][key], rtol=0, atol=1e-12)
    checkpoints = json.loads((root / "selected_checkpoints.json").read_text())
    for stage in checkpoints:
        require(file_sha256(root / f"{stage['stage']}_best.pt") == stage["checkpoint_sha256"],
                f"Checkpoint fingerprint mismatch: {stage['stage']}")
    report = {"passed": True, "method": provenance["method"], "n_prediction_files": len(paths),
        "n_points": actual["metrics"]["n_points"], "cohort_sha256": cohort.sha256,
        "metric_sha256": provenance["metric_sha256"], "selected_checkpoints_verified": len(checkpoints),
        "summary_metrics_recomputed": True, "per_event_metrics_recomputed": True}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
