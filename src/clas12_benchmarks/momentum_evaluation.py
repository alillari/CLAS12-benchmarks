"""Score keyed entrance momentum predictions against a frozen track manifest."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .data import file_sha256

AXES = ("px", "py", "pz")
# Saved companion labels underwent a float32 GPU reduction over padded sequences.
TRUTH_RTOL = 2e-6
TRUTH_ATOL_GEV = 1e-7


def _vector(row, prefix):
    return [float(row[f"{prefix}_{axis}_gev"] or "nan") for axis in AXES]


def _aligned_predictions(manifest, rows, path, companion):
    hdf5 = manifest.meta["schema"] == "clas12_momentum_tracks_v2"
    if hdf5 and companion:
        raise ValueError("Legacy companion predictions cannot be applied to an HDF5 target contract")
    key_to_local = {manifest.identity(i): j for j, i in enumerate(rows)}
    if len(key_to_local) != len(rows):
        raise ValueError("Manifest contains duplicate physical track identities")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = set(manifest.identity_fields)
        if hdf5:
            required.update({"manifest_sha256", "schema", "momentum_unit"})
        if companion:
            required.update({"n_hits", "source_event_index", "truth_segment_label"})
            required.update(f"{prefix}_{axis}_gev" for prefix in ("true", "cvt_entrance", "adapter") for axis in AXES)
            quantity, prefix = "vector", "adapter"
        else:
            vector_fields = {f"pred_{axis}_gev" for axis in AXES}
            if "pred_pt_gev" in columns and vector_fields & columns:
                raise ValueError("Ambiguous scalar and vector prediction columns")
            quantity, prefix = ("pt" if "pred_pt_gev" in columns else "vector"), "pred"
            required.update({"pred_pt_gev"} if quantity == "pt" else vector_fields)
        if not required <= columns or len(columns) != len(reader.fieldnames or []):
            raise ValueError(f"Missing or duplicate prediction columns: {required - columns}")
        pred = np.empty((len(rows), 3) if quantity == "vector" else len(rows))
        seen = np.zeros(len(rows), dtype=bool)
        if companion:
            archived_truth = np.empty_like(pred)
            archived_cvt = np.empty_like(pred)
        for line in reader:
            if hdf5:
                if (line["manifest_sha256"] != manifest.sha256
                        or line["schema"] != "clas12_momentum_predictions_v2" or line["momentum_unit"] != "GeV"):
                    raise ValueError("Prediction schema, units or manifest fingerprint mismatch")
                key = (line["source_manifest_id"], int(line["source_id"]),
                       int(line["original_ordinal"]), int(line["primary_tid"]))
            else:
                key = (line["source_file"], int(line["event"]), int(line["segment_label"]))
            j = key_to_local.get(key)
            if j is None or seen[j]:
                raise ValueError(f"Unexpected or duplicate prediction identity {key}")
            seen[j] = True
            pred[j] = _vector(line, prefix) if quantity == "vector" else float(line["pred_pt_gev"])
            if companion:
                i = rows[j]
                if (int(line["source_event_index"]) != manifest.keys[i, 0]
                        or int(line["truth_segment_label"]) != key[2]
                        or int(line["n_hits"]) != manifest.offsets[i+1] - manifest.offsets[i]):
                    raise ValueError(f"Companion track membership mismatch for {key}")
                archived_truth[j] = _vector(line, "true")
                archived_cvt[j] = _vector(line, "cvt_entrance")
        if not seen.all() or not np.isfinite(pred).all():
            raise ValueError(f"Missing {int((~seen).sum())} predictions or nonfinite model outputs")
    audit = {}
    if companion:
        for kind, saved, raw, rtol, atol in (
            ("truth", archived_truth, manifest.truth[rows], TRUTH_RTOL, TRUTH_ATOL_GEV),
            ("cvt", archived_cvt, manifest.cvt[rows], 1e-12, 1e-12),
        ):
            match = np.isclose(saved, raw, rtol=rtol, atol=atol, equal_nan=True)
            if not match.all():
                j = np.nonzero(~match)[0][0]
                raise ValueError(f"Archived {kind} disagrees with raw track {manifest.identity(rows[j])}: "
                                 f"{saved[j]} vs {raw[j]}")
            finite = np.isfinite(raw)
            audit[kind + "_max_abs_difference_gev"] = float(np.max(np.abs(saved[finite]-raw[finite]), initial=0))
    return quantity, pred, audit


def _pt(tracks):
    return np.hypot(tracks[:, 0], tracks[:, 1])


def _score(truth, predicted, quantity):
    n = len(truth)
    if not n:
        return {"n": 0}
    true_pt = _pt(truth)
    pred_pt = predicted if quantity == "pt" else _pt(predicted)
    pt_error = pred_pt - true_pt
    metrics = {"n": n, "pt_mae_mev": float(np.abs(pt_error).mean() * 1000),
               "pt_bias_mev": float(pt_error.mean() * 1000),
               "pt_rmse_mev": float(np.sqrt(np.mean(pt_error**2)) * 1000)}
    if quantity == "pt":
        return metrics
    true_p = np.linalg.norm(truth, axis=1)
    pred_p = np.linalg.norm(predicted, axis=1)
    error = pred_p - true_p
    valid = true_p > 1e-12
    relative = error[valid] / true_p[valid]
    phi_diff = np.degrees(np.arctan2(predicted[:, 1], predicted[:, 0]) - np.arctan2(truth[:, 1], truth[:, 0]))
    theta_diff = np.degrees(np.arctan2(pred_pt, predicted[:, 2]) - np.arctan2(true_pt, truth[:, 2]))
    component = np.abs(predicted-truth).mean(axis=0) * 1000
    metrics.update({f"{axis}_mae_mev": float(value) for axis, value in zip(AXES, component, strict=True)})
    metrics.update(component_mae_mev=float(component.mean()), p_mae_mev=float(np.abs(error).mean() * 1000),
                   p_bias_mev=float(error.mean() * 1000), theta_mae_deg=float(np.abs(theta_diff).mean()),
                   phi_mae_deg=float(np.abs((phi_diff + 180) % 360 - 180).mean()), n_relative=len(relative),
                   relative_bias_pct=float(relative.mean() * 100) if len(relative) else None,
                   relative_width68_pct=float(np.diff(np.quantile(relative, [.16, .84]))[0] * 50) if len(relative) else None,
                   tail10_pct=float(100 * np.mean(np.abs(relative) > .1)) if len(relative) else None,
                   tail25_pct=float(100 * np.mean(np.abs(relative) > .25)) if len(relative) else None)
    return metrics


def evaluate(manifest, path, role="evaluation", companion=False):
    path = Path(path)
    rows = manifest.rows(role)
    quantity, pred, audit = _aligned_predictions(manifest, rows, path, companion)
    truth, cvt = manifest.truth[rows], manifest.cvt[rows]
    common = np.isfinite(cvt).all(axis=1)
    reference = cvt[common] if quantity == "vector" else _pt(cvt[common])
    return {"schema": "clas12_momentum_scores_v2" if manifest.meta["schema"] == "clas12_momentum_tracks_v2" else "clas12_momentum_scores_v1", "manifest_sha256": manifest.sha256,
            "prediction_sha256": file_sha256(path), "role": role, "quantity": quantity,
            "n_missing_cvt": int((~common).sum()), "companion_row_audit": audit,
            "all_tracks": _score(truth, pred, quantity),
            "cvt_valid": {"model": _score(truth[common], pred[common], quantity),
                          "cvt": _score(truth[common], reference, quantity)}}


def verify_score_tables(result, directory, model):
    """Compare every published numeric score in both companion population tables."""
    if result.get("schema") == "clas12_momentum_scores_v2":
        raise ValueError("Legacy reference tables cannot verify the HDF5 target contract")
    if result["quantity"] != "vector" or result["role"] != "evaluation" or model == "cvt":
        raise ValueError("Companion comparison tables require vector predictions on the evaluation role")
    checks = {}
    for filename, name, actual in (
        ("metrics_all_50000_tracks.csv", model, result["all_tracks"]),
        ("metrics_common_population.csv", model, result["cvt_valid"]["model"]),
        ("metrics_common_population.csv", "cvt", result["cvt_valid"]["cvt"]),
    ):
        path = Path(directory) / filename
        with path.open(newline="") as stream:
            matches = [r for r in csv.DictReader(stream) if r["model"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one reference for model {name} in {filename}")
        fields = set(matches[0]) - {"model", "training_tracks"}
        required = {"n", "component_mae_mev", "p_mae_mev", "relative_width68_pct", "theta_mae_deg",
                    "phi_mae_deg", "tail10_pct", *(f"{axis}_mae_mev" for axis in AXES)}
        if not required <= fields:
            raise ValueError(f"Incomplete score table {filename}")
        for field in sorted(fields):
            expect = float(matches[0][field])
            value = actual[field]
            # Different float32 track reductions can move continuous scores slightly.
            # Population and tail counts must match the original table.
            discrete = field in ("n", "tail10_pct")
            if not np.isclose(value, expect, rtol=0 if discrete else 2e-6, atol=1e-12 if discrete else 2e-5):
                raise ValueError(f"{filename}/{name}/{field}: {value} vs {expect}")
        checks[f"{filename}/{name}"] = {"fields": sorted(fields), "table_sha256": file_sha256(path),
            "absolute_differences": {field: abs(actual[field] - float(matches[0][field])) for field in sorted(fields)}}
    return checks


def verify_native_pt(result, summary_path):
    """Check the native evaluator's pT scores, which the campaign CSVs omit."""
    if result.get("schema") == "clas12_momentum_scores_v2":
        raise ValueError("Legacy reference summaries cannot verify the HDF5 target contract")
    summary = json.loads(Path(summary_path).read_text())
    if summary["evaluation_contract"] != "clas12_momentum_entrance_v1" or result["role"] != "evaluation":
        raise ValueError("Expected companion entrance evaluation")
    differences = {}
    for method, scores in (("adapter", result["all_tracks"]), ("cvt", result["cvt_valid"]["cvt"])):
        native = summary["methods"][method]
        if scores["n"] != native["n"]:
            raise ValueError(f"Native pT denominator differs for {method}")
        if not scores["n"]:
            continue
        for metric in ("mae", "rmse"):
            measured = scores[f"pt_{metric}_mev"]
            expected = native["pt"][f"{metric}_gev"] * 1000
            if not np.isclose(measured, expected, rtol=2e-6, atol=2e-5):
                raise ValueError(f"Native {method} pT {metric} differs: {measured} vs {expected}")
            differences[f"{method}_{metric}_mev"] = abs(measured - expected)
    return {"summary_sha256": file_sha256(summary_path), "absolute_differences": differences}
