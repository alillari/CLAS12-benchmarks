"""CPU rescoring of saved, final benchmark assignments; never run inference."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import scipy
import sklearn

from .data import Cohort, file_sha256
from .finding_evaluation import (DEFAULT_METRICS, FindingEvaluation, PopulationFingerprint, contract,
                                 finding_fingerprints, finding_sequence, load_metrics,
                                 reference_receipt, verify_evaluation_metadata)
from .predictions import prediction_schema
from .prepare_cohort import write_json


def saved_predictions(parent, cohort, provenance, events):
    paths = sorted((parent / "predictions").glob("*.npz"))
    if len(paths) != len(events):
        raise ValueError("Prediction event count mismatch")
    for index, (path, event) in enumerate(zip(paths, events, strict=True)):
        if path.name != f"event_{index:06d}.npz":
            raise ValueError("Prediction filename/order mismatch")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            expected = {"schema": prediction_schema(event.identity), "identity": event.identity,
                        "cohort_sha256": cohort.sha256, "method": provenance["method"]}
            if any(metadata.get(k) != value for k, value in expected.items()):
                raise ValueError(f"Prediction event identity/cohort/method mismatch: {path.name}")
            if provenance.get("finding_evaluation") is not None:
                if metadata.get("finding_evaluation") != provenance["finding_evaluation"]:
                    raise ValueError("Prediction evaluation contract mismatch")
            if (saved["original_point_index"].dtype.kind not in "iu" or
                    not np.array_equal(saved["original_point_index"], event.original_point_index)):
                raise ValueError("Prediction original point identity mismatch")
            prediction = saved["predictions"]
            if prediction.shape != (len(event.raw_features),) or prediction.dtype.kind not in "iu":
                raise ValueError("Invalid saved complete-event assignments")
            yield prediction


def parent_receipt(parent):
    names = ["provenance.json", "selected_checkpoints.json", "frozen_selection.json", "track_finding_metrics.py"]
    names.extend(name for name in ("summary.json", "per_event_metrics.jsonl", "per_event_coatjava_metrics.jsonl")
                 if (parent / name).is_file())
    selected = json.loads((parent / "selected_checkpoints.json").read_text())
    if len({row["stage"] for row in selected}) != len(selected):
        raise ValueError("Duplicate selected checkpoint stages")
    for row in selected:
        if row["stage"] not in {"embedding", "filter", "gnn"}:
            raise ValueError("Unknown selected checkpoint stage")
        path = parent / f"{row['stage']}_best.pt"
        if file_sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint fingerprint mismatch")
        names.append(path.name)
    names.extend(str(p.relative_to(parent)) for p in sorted((parent / "predictions").glob("*.npz")))
    return {name: file_sha256(parent / name) for name in names}


def rescore(parent, cohort_path, output, *, metrics_path=DEFAULT_METRICS, settings=None):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    # A distinct new directory is mandatory, including for interrupted attempts.
    # Reject a nested output as it would modify the archived parent result tree.
    if output == parent or output.is_relative_to(parent):
        raise ValueError("Rescoring output must be outside the parent result directory")
    if output.exists():
        raise FileExistsError(output)
    settings = contract(settings, new_run=True)
    metrics_sha256 = file_sha256(metrics_path)
    metrics = load_metrics(metrics_path)
    evaluation = FindingEvaluation(metrics, settings)
    provenance = json.loads((parent / "provenance.json").read_text())
    before = parent_receipt(parent)
    if before["track_finding_metrics.py"] != provenance["metric_sha256"]:
        raise ValueError("Parent evaluator fingerprint mismatch")
    with Cohort(cohort_path) as cohort:
        if cohort.sha256 != provenance["cohort_sha256"]:
            raise ValueError("Parent cohort fingerprint mismatch")
        if cohort.data_fingerprints() != provenance["data_array_sha256"]:
            raise ValueError("Parent consumed data fingerprint mismatch")
        if "finding_data_sha256" in provenance:
            original = FindingEvaluation(settings=provenance["finding_evaluation"])
            if finding_fingerprints(cohort, original) != provenance["finding_data_sha256"]:
                raise ValueError("Parent finding provenance sidecars changed")
        sidecars = finding_fingerprints(cohort, evaluation)
        events = finding_sequence(cohort, "test", evaluation)
        output.mkdir(parents=True)
        shutil.copyfile(metrics_path, output / "track_finding_metrics.py")
        shutil.copyfile(DEFAULT_METRICS.parent / "track_eligibility.py", output / "track_eligibility.py")
        if file_sha256(output / "track_finding_metrics.py") != metrics_sha256:
            raise ValueError("Scorer changed while its snapshot was being copied")
        evaluation = FindingEvaluation(load_metrics(output / "track_finding_metrics.py"), settings)
        package = Path(__file__).parent
        source_files = ["rescore_finding.py", "finding_evaluation.py", "data.py", "sequences.py",
                        "predictions.py", "prepare_cohort.py", "hdf5.py" if cohort.is_hdf5 else "sources.py"]
        source_hashes = {name: file_sha256(package / name) for name in source_files}
        # Do not duplicate checkpoints or predictions. Score only, preserving
        # their hashes and pointers in the receipt.
        predictions = saved_predictions(parent, cohort, provenance, events)
        rows, baseline, relabelled = [], [], 0
        population = PopulationFingerprint()
        for index, (event, prediction) in enumerate(zip(events, predictions, strict=True)):
            raw = cohort.event("test", index)
            population.add(event, evaluation)
            relabelled += int(np.count_nonzero(raw.labels["seg_target"] != event.labels["seg_target"]))
            metadata = {"identity": event.identity, "finding_evaluation": settings}
            rows.append({**metadata, **evaluation.event_metrics(event, prediction)})
            baseline.append({**metadata, **evaluation.event_metrics(event, event.labels["coatjava_seg_pred"])})
        summary = {**evaluation.metadata(), "metric_view": "native", "matching": settings["matching"],
                   "evaluation_population_sha256": population.hexdigest(),
                   "metrics": evaluation.summarize(rows), "coatjava": evaluation.summarize(baseline)}
        summary["model_minus_coatjava"] = evaluation.compare(summary["metrics"], summary["coatjava"])
        verify_evaluation_metadata(summary, evaluation)
        if (parent_receipt(parent) != before or finding_fingerprints(cohort, evaluation) != sidecars or
                cohort.data_fingerprints() != provenance["data_array_sha256"] or
                file_sha256(metrics_path) != metrics_sha256 or
                any(file_sha256(package / name) != digest for name, digest in source_hashes.items())):
            raise ValueError("Parent artifacts or provenance changed during rescoring")
        receipt = {"schema": "clas12_finding_rescore_v1", "parent_run": str(parent),
                   "parent_artifact_sha256": before, "cohort_sha256": cohort.sha256,
                   "metrics_source_sha256": file_sha256(metrics_path), "metrics_source": str(metrics_path),
                   "scorer_source_sha256": file_sha256(__file__), "fm_reference": reference_receipt(),
                   "evaluation_source_sha256": source_hashes,
                   "finding_evaluation": settings, "finding_data_sha256": sidecars,
                   "training_finding_evaluation": provenance.get("finding_evaluation"),
                   "n_relabelled_truth_points": relabelled, "full_event_membership_verified": True,
                   "checkpoint_selection_modified": False, "thresholds_recalibrated": False,
                   "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                   "sklearn_version": sklearn.__version__}
        for name, records in (("per_event_metrics.jsonl", rows), ("per_event_coatjava_metrics.jsonl", baseline)):
            with (output / name).open("x") as stream:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "receipt.json", receipt)
        write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--truth-policy", choices=("as_saved", "primary_mc_min4_v1"), default="primary_mc_min4_v1")
    parser.add_argument("--min-purity", type=float, default=.5)
    parser.add_argument("--min-efficiency", type=float, default=.5)
    parser.add_argument("--iou-threshold", type=float, default=.5)
    args = parser.parse_args()
    result = rescore(args.run, args.cohort, args.output, metrics_path=args.metrics,
                     settings={"profile": "mc_reconstruction_v1", "truth_policy": args.truth_policy,
                               "matching": {"min_purity": args.min_purity,
                                            "min_efficiency": args.min_efficiency, "iou_threshold": args.iou_threshold}})
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
