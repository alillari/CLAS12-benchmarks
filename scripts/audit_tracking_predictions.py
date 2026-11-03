"""Recompute shared metrics from every saved point prediction after a run."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.prepare_cohort import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    cohort = Cohort(args.cohort)
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["cohort_sha256"] == cohort.sha256
    metric_path = root / "track_finding_metrics.py"
    assert file_sha256(metric_path) == provenance["metric_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_tracking_metric_audit", metric_path)
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    paths = sorted((root / "predictions").glob("*.npz"))
    assert len(paths) == len(cohort.roles["test"])
    rows, baseline = [], []
    for i, path in enumerate(paths):
        assert path.name == f"event_{i:06d}.npz"
        event = cohort.event("test", i)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            assert metadata["schema"] == "clas12_baseline_predictions_v1"
            assert metadata["cohort_sha256"] == cohort.sha256
            assert metadata["identity"] == event.identity
            assert metadata["method"] == provenance["method"]
            prediction = saved["predictions"]
            assert prediction.shape == (len(event.positions),)
            assert np.issubdtype(prediction.dtype, np.integer)
            np.testing.assert_array_equal(saved["original_point_index"], event.original_point_index)
            rows.append(metrics.event_track_metrics(event.labels["seg_target"], prediction))
            baseline.append(metrics.event_track_metrics(event.labels["seg_target"], event.labels["coatjava_seg_pred"]))
    actual = {"metrics": metrics.summarize_event_metrics(rows), "coatjava": metrics.summarize_event_metrics(baseline)}
    expected = json.loads((root / "summary.json").read_text())
    for group, values in actual.items():
        for key, value in values.items():
            if value is None:
                assert expected[group][key] is None
            else:
                np.testing.assert_allclose(value, expected[group][key], rtol=0, atol=1e-12)
    with (root / "per_event_metrics.jsonl").open() as stream:
        assert sum(1 for _ in stream) == len(paths)
    checkpoints = json.loads((root / "selected_checkpoints.json").read_text())
    for stage in checkpoints:
        assert file_sha256(root / f"{stage['stage']}_best.pt") == stage["checkpoint_sha256"]
    report = {"passed": True, "method": provenance["method"], "n_prediction_files": len(paths),
        "n_points": actual["metrics"]["n_points"], "cohort_sha256": cohort.sha256,
        "metric_sha256": provenance["metric_sha256"], "selected_checkpoints_verified": len(checkpoints),
        "summary_metrics_recomputed": True}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
