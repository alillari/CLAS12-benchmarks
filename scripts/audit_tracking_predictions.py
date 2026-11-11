"""Recompute shared metrics from every saved point prediction after a run."""
import argparse
import json
from pathlib import Path

import numpy as np

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.predictions import prediction_schema
from clas12_benchmarks.prepare_cohort import write_json
from clas12_benchmarks.finding_evaluation import (FindingEvaluation, PopulationFingerprint, finding_sequence,
    finding_fingerprints, load_metrics, verify_evaluation_metadata, verify_reference_provenance)


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
    with Cohort(args.cohort) as cohort:
        provenance = json.loads((root / "provenance.json").read_text())
        verify_reference_provenance(provenance)
        require(provenance["cohort_sha256"] == cohort.sha256, "Cohort fingerprint mismatch")
        if "data_array_sha256" in provenance:
            require(cohort.data_fingerprints() == provenance["data_array_sha256"],
                    "Consumed data fingerprint mismatch")
        metric_path = root / "track_finding_metrics.py"
        require(file_sha256(metric_path) == provenance["metric_sha256"], "Evaluator fingerprint mismatch")
        metrics = load_metrics(metric_path)
        evaluation = FindingEvaluation(metrics, provenance.get("finding_evaluation"))
        explicit = "finding_evaluation" in provenance
        if explicit:
            require(provenance["config"]["finding_evaluation"] == evaluation.contract,
                    "Training/evaluation contract mismatch")
            require(provenance.get("finding_data_sha256") == finding_fingerprints(cohort, evaluation),
                    "Consumed finding provenance fingerprint mismatch")
        events = finding_sequence(cohort, "test", evaluation)
        population = PopulationFingerprint()
        paths = sorted((root / "predictions").glob("*.npz"))
        require(len(paths) == len(cohort.roles["test"]), "Prediction event count mismatch")
        with (root / "per_event_metrics.jsonl").open() as stream:
            cached_rows = [json.loads(line) for line in stream]
        require(len(cached_rows) == len(paths), "Per-event metric count mismatch")
        rows, baseline = [], []
        for i, path in enumerate(paths):
            require(path.name == f"event_{i:06d}.npz", "Prediction filename/order mismatch")
            event = events[i]
            if explicit:
                population.add(event, evaluation)
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                require(metadata["schema"] == prediction_schema(event.identity), "Prediction schema mismatch")
                require(metadata["cohort_sha256"] == cohort.sha256, "Prediction cohort mismatch")
                require(metadata["identity"] == event.identity, "Prediction event identity mismatch")
                require(metadata["method"] == provenance["method"], "Prediction method mismatch")
                if explicit:
                    require(metadata.get("finding_evaluation") == evaluation.contract, "Prediction evaluation contract mismatch")
                prediction = saved["predictions"]
                require(prediction.shape == (len(event.positions),), "Prediction point count mismatch")
                require(np.issubdtype(prediction.dtype, np.integer), "Prediction labels must be integers")
                np.testing.assert_array_equal(saved["original_point_index"], event.original_point_index)
                row = evaluation.event_metrics(event, prediction)
                identity = {"identity": event.identity}
                if explicit:
                    identity["finding_evaluation"] = evaluation.contract
                require(cached_rows[i] == {**identity, **row},
                        f"Per-event metrics or identity differ from predictions: event {i}")
                rows.append(row)
                if "coatjava_seg_pred" in event.labels:
                    baseline.append({**identity, **evaluation.event_metrics(event, event.labels["coatjava_seg_pred"])})
        require(not baseline or len(baseline) == len(rows), "COATJAVA event population mismatch")
        actual = {"metrics": evaluation.summarize(rows), "coatjava": evaluation.summarize(baseline)}
        expected = json.loads((root / "summary.json").read_text())
        if explicit:
            verify_evaluation_metadata(expected, evaluation)
            require(expected.get("evaluation_population_sha256") == population.hexdigest(), "Evaluation population fingerprint mismatch")
            with (root / "per_event_coatjava_metrics.jsonl").open() as stream:
                require([json.loads(line) for line in stream] == baseline, "COATJAVA per-event cache mismatch")
        def compare(value, expected_value, label):
            if isinstance(value, dict):
                require(isinstance(expected_value, dict) and value.keys() == expected_value.keys(),
                        f"Summary metric keys mismatch: {label}")
                for key in value:
                    compare(value[key], expected_value[key], f"{label}/{key}")
            elif value is None or isinstance(value, (str, list)):
                require(value == expected_value, f"Undefined summary metric mismatch: {label}")
            else:
                np.testing.assert_allclose(value, expected_value, rtol=0, atol=1e-12)
        for group, values in actual.items():
            for key, value in values.items():
                compare(value, expected[group][key], f"{group}/{key}")
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
