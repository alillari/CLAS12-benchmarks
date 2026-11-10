"""Versioned track-finding protocol; no model or trainer dependency.

Missing contracts mean historical scoring. New runs opt into the committed FM
profile explicitly. Raw cohorts, noise/PID labels and saved predictions are never
rewritten by the finding-only truth projection.
"""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from .data import event_key, file_sha256
from .fm_reference import track_finding_metrics as reference
from .fm_reference.track_eligibility import eligible_primary_labels, primary_track_eligibility_contract
from .sequences import CachedSequence


MC_PROFILE = reference.MC_RECONSTRUCTION_VERSION
TRUTH_POLICY = primary_track_eligibility_contract()["policy"]
DEFAULT_METRICS = Path(reference.__file__)
PROVENANCE_FIELDS = tuple("clusters." + key for key in
                          ("background_only", "ownership", "origin_noise", "purity"))
MC_HEADLINES = ("mc_track_dice_mean", "mc_hit_completeness_global", "mc_hit_purity_global",
                "mc_hit_f1_global", "mc_bg_contamination_global", "bg_leakage_to_mc_global")


def contract(value=None, *, new_run=False):
    """Normalize an explicit contract, retaining legacy defaults for old records."""
    value = dict(value or {})
    unknown = value.keys() - {"profile", "truth_policy", "matching"}
    if unknown:
        raise ValueError(f"Unknown finding evaluation settings: {sorted(unknown)}")
    profile = value.get("profile", MC_PROFILE if new_run else "legacy")
    policy = value.get("truth_policy", TRUTH_POLICY if new_run else "as_saved")
    if profile not in {"legacy", MC_PROFILE} or policy not in {"as_saved", TRUTH_POLICY}:
        raise ValueError("Unsupported finding metric profile or truth policy")
    matching = {**vars(reference.MatchConfig()), **value.get("matching", {})}
    if matching.keys() != vars(reference.MatchConfig()).keys():
        raise ValueError("Unknown accepted-match settings")
    if matching["background_label"] != -1:
        raise ValueError("CLAS12 finding truth requires background_label=-1")
    for key in ("iou_threshold", "min_purity", "min_efficiency"):
        if not np.isfinite(matching[key]) or not 0 <= matching[key] <= 1:
            raise ValueError(f"Invalid accepted-match cut: {key}")
    return {"profile": profile, "truth_policy": policy, "matching": matching}


def load_metrics(path=DEFAULT_METRICS):
    """Execute the exact source bytes without reading or writing bytecode caches."""
    path = Path(path).resolve()
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location("clas12_finding_metrics_" + hashlib.sha256(source).hexdigest(), path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module


class FindingEvaluation:
    def __init__(self, metrics=reference, settings=None):
        self.metrics = metrics
        self.contract = contract(settings)
        if settings is None and hasattr(metrics, "MatchConfig"):
            # Preserve defaults embedded in archived evaluator snapshots.
            self.contract = contract({"matching": vars(metrics.MatchConfig())})
        self.mc = self.contract["profile"] == MC_PROFILE
        if self.mc and (getattr(metrics, "MC_RECONSTRUCTION_VERSION", None) != MC_PROFILE or
                        not hasattr(metrics, "MCReconstructionAccumulator")):
            raise ValueError("MC reconstruction requires a compatible versioned FM evaluator")

    def metadata(self):
        policy = self.contract["truth_policy"]
        return {"finding_evaluation": self.contract,
                "finding_metrics_version": 2,
                "metrics_source_sha256": file_sha256(self.metrics.__file__),
                "eligibility_source_sha256": (file_sha256(DEFAULT_METRICS.parent / "track_eligibility.py")
                                              if policy == TRUTH_POLICY else None),
                "track_eligibility_contract": primary_track_eligibility_contract() if policy == TRUTH_POLICY else None,
                **({"mc_reconstruction_version": MC_PROFILE,
                    "primary_evaluation_metric": "mc_track_dice_mean"} if self.mc else {})}

    def prepare_event(self, event):
        if self.contract["truth_policy"] == "as_saved":
            return event
        labels = dict(event.labels)
        labels["seg_target"] = eligible_primary_labels(labels["seg_target"])
        return replace(event, labels=labels)

    def event_metrics(self, event, prediction):
        truth = self.prepare_event(event).labels["seg_target"]
        if not self.mc:
            if hasattr(self.metrics, "MatchConfig"):
                return self.metrics.event_track_metrics(truth, prediction,
                             config=self.metrics.MatchConfig(**self.contract["matching"]))
            # Historical test/plugins sometimes expose only two positional args.
            if self.contract["matching"] != vars(reference.MatchConfig()):
                raise ValueError("The legacy evaluator cannot configure accepted-match cuts")
            return self.metrics.event_track_metrics(truth, prediction)
        context = event.context or {}
        raw = {key.rsplit(".", 1)[1]: context[key] for key in PROVENANCE_FIELDS if key in context}
        provenance = None
        if "background_only" in raw:
            provenance = self.metrics.cluster_provenance_from_hdf5(**raw)
        return self.metrics.event_mc_reconstruction_metrics(
            truth, prediction, point_ids=event.original_point_index, provenance=provenance,
            config=self.metrics.MatchConfig(**self.contract["matching"]))

    def summarize(self, rows):
        if not self.mc:
            return self.metrics.summarize_event_metrics(rows)
        result = self.metrics.summarize_mc_reconstruction_metrics(rows)
        # Derived from sufficient counts: remains zero for all-missed truth,
        # even though hit purity is undefined when no prediction is associated.
        denominator = result["n_signal_points"] + result["n_mc_associated_pred_points"]
        result["mc_hit_f1_global"] = 2 * result["n_mc_correct_points"] / denominator if denominator else None
        return result

    def compare(self, candidate, baseline):
        names = (*self.metrics.COMPARISON_METRICS, *MC_HEADLINES) if self.mc else self.metrics.COMPARISON_METRICS
        return self.metrics.compare_metric_summaries(candidate, baseline, names)


def finding_sequence(cohort, role, evaluation, count=None, cache_size=8):
    count = len(cohort.roles[role]) if count is None else count
    def read(index):
        event = cohort.event(role, index)
        if evaluation.mc and cohort.is_hdf5:
            member = cohort.roles[role][index]
            source = cohort.sources[member["dataset"]]
            fields = [key for key in PROVENANCE_FIELDS
                      if key.split(".")[1] in source.schema["fields"].get("clusters", {})]
            # Without background_only none of the partial sidecars establish
            # the complete provenance contract; report unknown instead.
            if "clusters.background_only" in fields:
                context = {**(event.context or {}), **source.read(member["index"], fields)}
                event = replace(event, context=context)
        return evaluation.prepare_event(event)
    return CachedSequence(count, read, cache_size)


def finding_fingerprints(cohort, evaluation):
    """Additional consumed sidecars; never change old cohort content digests."""
    if not evaluation.mc or not cohort.is_hdf5:
        return {}
    result = {}
    for name, (_, rows, _) in cohort.datasets.items():
        source = cohort.sources[name]
        fields = [key for key in PROVENANCE_FIELDS
                  if key.split(".")[1] in source.schema["fields"].get("clusters", {})]
        if "clusters.background_only" not in fields:
            fields = []
        result[name] = {"fields": fields, "sha256": source.content_digest(rows, fields=fields)}
    return result


def require_same_contract(left, right):
    if contract(left.get("finding_evaluation")) != contract(right.get("finding_evaluation")):
        raise ValueError("Finding evaluation contracts differ; rescore both with the same protocol")


def verify_evaluation_metadata(summary, evaluation):
    """Do not accept an old score merely because a caller attached a version."""
    for key, expected in evaluation.metadata().items():
        if summary.get(key) != expected:
            raise ValueError(f"Finding evaluation metadata mismatch: {key}")


class PopulationFingerprint:
    """Physical cluster/truth membership, independent of model normalization."""

    def __init__(self):
        self.digest = hashlib.sha256()

    def add(self, event, evaluation):
        from .hdf5 import update_digest
        self.digest.update(json.dumps(event_key(event.identity)).encode())
        arrays = {"features": event.raw_features, "point_ids": event.original_point_index,
                  "truth": evaluation.prepare_event(event).labels["seg_target"]}
        if "coatjava_seg_pred" in event.labels:
            arrays["coatjava"] = event.labels["coatjava_seg_pred"]
        for field, value in arrays.items():
            update_digest(self.digest, field, value)
        for field in PROVENANCE_FIELDS:
            if field in (event.context or {}):
                update_digest(self.digest, field, event.context[field])

    def hexdigest(self):
        return self.digest.hexdigest()


def require_same_population(left, right):
    require_same_contract(left, right)
    a, b = left.get("evaluation_population_sha256"), right.get("evaluation_population_sha256")
    versioned = "finding_evaluation" in left or "finding_evaluation" in right
    if (versioned or a is not None or b is not None) and (not a or a != b):
        raise ValueError("Finding evaluation event/cluster populations differ")
    for key in ("metrics_source_sha256", "eligibility_source_sha256"):
        a, b = left.get(key), right.get(key)
        required = versioned and (key == "metrics_source_sha256" or
                                 contract(left.get("finding_evaluation"))["truth_policy"] == TRUTH_POLICY)
        if (required and not a) or a != b:
            raise ValueError(f"Finding evaluator source fingerprints differ or are missing: {key}")


def reference_receipt():
    directory = Path(reference.__file__).parent
    receipt = json.loads((directory / "sources.json").read_text())
    for name, item in receipt["files"].items():
        if file_sha256(directory / name) != item["sha256"]:
            raise ValueError(f"Pinned FM reference changed: {name}")
    return receipt


def verify_reference_provenance(provenance):
    """Frozen inference/audits must retain the recorded eligibility definition."""
    if "finding_evaluation" in provenance and provenance.get("fm_reference") != reference_receipt():
        raise ValueError("Frozen FM reference differs or is missing; use the matching source snapshot")


def evaluate_events(events, predictions, metrics=reference, *, settings=None,
                    output=None, method=None, cohort_sha=None):
    """Score the same complete-event labels for the model and stored COATJAVA."""
    from .predictions import save_predictions
    from .prepare_cohort import write_json
    evaluation = FindingEvaluation(metrics, settings)
    population = PopulationFingerprint()
    rows, baseline = [], []
    for i, (event, prediction) in enumerate(zip(events, predictions, strict=True)):
        prediction = np.asarray(prediction)
        if prediction.shape != (len(event.raw_features),) or prediction.dtype.kind not in "iu":
            raise ValueError("Invalid point prediction mapping")
        if settings is not None:
            population.add(event, evaluation)
        metadata = {"identity": event.identity}
        if settings is not None:
            metadata["finding_evaluation"] = evaluation.contract
        rows.append({**metadata, **evaluation.event_metrics(event, prediction)})
        if "coatjava_seg_pred" in event.labels:
            baseline.append({**metadata, **evaluation.event_metrics(event, event.labels["coatjava_seg_pred"])})
        if output:
            save_predictions(Path(output) / "predictions" / f"event_{i:06d}.npz", event, prediction,
                event.original_point_index, method=method, cohort_sha256=cohort_sha,
                finding_evaluation=evaluation.contract if settings is not None else None)
    if baseline and len(baseline) != len(rows):
        raise ValueError("COATJAVA must cover the same complete event population")
    summary = evaluation.summarize(rows)
    result = {"metrics": summary, "metric_view": "native",
              "matching": evaluation.contract["matching"], "coatjava": evaluation.summarize(baseline)}
    if settings is not None:
        result.update(evaluation.metadata())
        result["evaluation_population_sha256"] = population.hexdigest()
    if baseline:
        result["model_minus_coatjava"] = evaluation.compare(summary, result["coatjava"])
    if output:
        output = Path(output)
        write_json(output / "summary.json", result)
        groups = {"per_event_metrics.jsonl": rows}
        if settings is not None:
            groups["per_event_coatjava_metrics.jsonl"] = baseline
        for name, records in groups.items():
            with (output / name).open("w") as stream:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    return result
