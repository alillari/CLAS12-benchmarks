"""Metrics for event-level track-finding segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class MatchConfig:
    background_label: int = -1
    iou_threshold: float = 0.5
    min_purity: float = 0.5
    min_efficiency: float = 0.5


COMPARISON_METRICS = (
    "ari_signal",
    "ari_with_background",
    "track_efficiency_global",
    "track_purity_global",
    "matched_iou_mean",
    "matched_purity_mean",
    "matched_efficiency_mean",
    "fake_rate",
    "miss_rate",
    "split_rate",
    "merge_rate",
    "background_rejection",
    "background_contamination",
    "signal_loss_to_background",
)


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def compare_metric_summaries(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metric_names: tuple[str, ...] = COMPARISON_METRICS,
) -> dict[str, float | None]:
    """Return candidate-minus-baseline deltas for shared headline metrics."""
    deltas: dict[str, float | None] = {}
    for name in metric_names:
        candidate_value = candidate.get(name)
        baseline_value = baseline.get(name)
        if candidate_value is None or baseline_value is None:
            deltas[name] = None
        else:
            deltas[name] = finite_or_none(float(candidate_value) - float(baseline_value))
    return deltas


def event_track_metrics(
    truth_labels: np.ndarray,
    pred_labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    pred_signal_mask: np.ndarray | None = None,
    config: MatchConfig | None = None,
) -> dict[str, Any]:
    """Compute one event's clustering and matched-track metrics.

    ``truth_labels`` uses ``config.background_label`` for background/no segment.
    ``pred_labels`` uses arbitrary predicted instance ids; values < 0 are
    treated as predicted background.
    """

    config = config or MatchConfig()
    truth = np.asarray(truth_labels).reshape(-1)
    pred = np.asarray(pred_labels).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"truth/pred shape mismatch: {truth.shape} vs {pred.shape}")

    if valid_mask is None:
        valid = np.ones(truth.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask).reshape(-1).astype(bool)
    if pred_signal_mask is None:
        pred_signal = pred >= 0
    else:
        pred_signal = np.asarray(pred_signal_mask).reshape(-1).astype(bool) & (pred >= 0)

    truth_background = valid & (truth == config.background_label)
    truth_signal = valid & ~truth_background
    pred_signal = pred_signal & valid
    pred = np.where(pred_signal, pred, -1)

    true_ids = sorted(np.unique(truth[truth_signal]).tolist())
    pred_ids = sorted(np.unique(pred[pred_signal]).tolist())
    pred_ids_on_truth_signal = sorted(np.unique(pred[truth_signal]).tolist())

    if truth_signal.sum() > 1 and len(true_ids) > 1:
        ari_signal = float(adjusted_rand_score(truth[truth_signal], pred[truth_signal]))
    elif truth_signal.sum() > 0:
        # A separate predicted noise instance outside the true signal support
        # must not penalize a one-track signal event.
        ari_signal = 1.0 if len(pred_ids_on_truth_signal) <= 1 else 0.0
    else:
        ari_signal = None

    if valid.sum() > 1 and len(np.unique(truth[valid])) > 1:
        pred_with_bg = pred.copy()
        pred_with_bg[~pred_signal] = -1
        ari_with_background = float(adjusted_rand_score(truth[valid], pred_with_bg[valid]))
    else:
        ari_with_background = None

    n_true = len(true_ids)
    n_pred = len(pred_ids)
    iou = np.zeros((n_pred, n_true), dtype=np.float64)
    purity = np.zeros_like(iou)
    efficiency = np.zeros_like(iou)

    for pidx, pred_id in enumerate(pred_ids):
        pred_points = pred_signal & (pred == pred_id)
        pred_count = int(pred_points.sum())
        for tidx, true_id in enumerate(true_ids):
            true_points = truth_signal & (truth == true_id)
            true_count = int(true_points.sum())
            intersection = int((pred_points & true_points).sum())
            union = int((pred_points | true_points).sum())
            iou[pidx, tidx] = _safe_div(intersection, union) or 0.0
            purity[pidx, tidx] = _safe_div(intersection, pred_count) or 0.0
            efficiency[pidx, tidx] = _safe_div(intersection, true_count) or 0.0

    matches = []
    if n_pred and n_true:
        eligible = ((iou >= config.iou_threshold) & (purity >= config.min_purity)
                    & (efficiency >= config.min_efficiency))
        # One more eligible match outweighs the largest possible total IoU gain.
        bonus = min(n_pred, n_true) + 1
        weights = np.where(eligible, bonus + iou, 0.)
        row_ind, col_ind = linear_sum_assignment(-weights)
        for pidx, tidx in zip(row_ind, col_ind):
            if eligible[pidx, tidx]:
                matches.append({
                    "pred_id": int(pred_ids[pidx]),
                    "true_id": int(true_ids[tidx]),
                    "iou": float(iou[pidx, tidx]),
                    "purity": float(purity[pidx, tidx]),
                    "efficiency": float(efficiency[pidx, tidx]),
                })

    matched_true = {match["true_id"] for match in matches}
    matched_pred = {match["pred_id"] for match in matches}
    split_truth = 0
    for true_id in true_ids:
        overlaps = [
            pred_id for pred_id in pred_ids
            if (pred_signal & (pred == pred_id) & truth_signal & (truth == true_id)).sum() > 0
        ]
        if len(overlaps) > 1:
            split_truth += 1
    merged_pred = 0
    for pred_id in pred_ids:
        overlaps = np.unique(truth[pred_signal & (pred == pred_id) & truth_signal])
        if len(overlaps) > 1:
            merged_pred += 1

    background_pred_background = int((truth_background & ~pred_signal).sum())
    background_as_signal = int((truth_background & pred_signal).sum())
    signal_as_background = int((truth_signal & ~pred_signal).sum())

    return {
        "n_points": int(valid.sum()),
        "n_signal_points": int(truth_signal.sum()),
        "n_background_points": int(truth_background.sum()),
        "n_true_tracks": int(n_true),
        "n_pred_tracks": int(n_pred),
        "n_matched_tracks": int(len(matches)),
        "ari_signal": finite_or_none(ari_signal),
        "ari_with_background": finite_or_none(ari_with_background),
        "track_efficiency": _safe_div(len(matched_true), n_true),
        "track_purity": _safe_div(len(matched_pred), n_pred),
        "matched_iou_mean": _mean_or_none([m["iou"] for m in matches]),
        "matched_purity_mean": _mean_or_none([m["purity"] for m in matches]),
        "matched_efficiency_mean": _mean_or_none([m["efficiency"] for m in matches]),
        "fake_rate": _safe_div(n_pred - len(matched_pred), n_pred),
        "miss_rate": _safe_div(n_true - len(matched_true), n_true),
        "split_rate": _safe_div(split_truth, n_true),
        "merge_rate": _safe_div(merged_pred, n_pred),
        "background_rejection": _safe_div(background_pred_background, int(truth_background.sum())),
        "background_contamination": _safe_div(background_as_signal, int(pred_signal.sum())),
        "signal_loss_to_background": _safe_div(signal_as_background, int(truth_signal.sum())),
        "matches": matches,
    }


def summarize_event_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {
        "n_events": len(rows),
        "n_points": int(sum(row.get("n_points", 0) for row in rows)),
        "n_signal_points": int(sum(row.get("n_signal_points", 0) for row in rows)),
        "n_background_points": int(sum(row.get("n_background_points", 0) for row in rows)),
        "n_true_tracks": int(sum(row.get("n_true_tracks", 0) for row in rows)),
        "n_pred_tracks": int(sum(row.get("n_pred_tracks", 0) for row in rows)),
        "n_matched_tracks": int(sum(row.get("n_matched_tracks", 0) for row in rows)),
    }
    for key in (
        "ari_signal",
        "ari_with_background",
        "track_efficiency",
        "track_purity",
        "matched_iou_mean",
        "matched_purity_mean",
        "matched_efficiency_mean",
        "fake_rate",
        "miss_rate",
        "split_rate",
        "merge_rate",
        "background_rejection",
        "background_contamination",
        "signal_loss_to_background",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = finite_or_none(float(np.mean(values))) if values else None
    summary["track_efficiency_global"] = _safe_div(summary["n_matched_tracks"], summary["n_true_tracks"])
    summary["track_purity_global"] = _safe_div(summary["n_matched_tracks"], summary["n_pred_tracks"])
    return summary


EVENT_MEAN_METRICS = (
    "ari_signal",
    "ari_with_background",
    "track_efficiency",
    "track_purity",
    "matched_iou_mean",
    "matched_purity_mean",
    "matched_efficiency_mean",
    "fake_rate",
    "miss_rate",
    "split_rate",
    "merge_rate",
    "background_rejection",
    "background_contamination",
    "signal_loss_to_background",
)
EVENT_COUNT_METRICS = (
    "n_points",
    "n_signal_points",
    "n_background_points",
    "n_true_tracks",
    "n_pred_tracks",
    "n_matched_tracks",
)


class EventMetricAccumulator:
    """Streaming equivalent of ``summarize_event_metrics`` for validation and evaluation."""

    def __init__(self) -> None:
        self.n_events = 0
        self.counts = {key: 0 for key in EVENT_COUNT_METRICS}
        self.sums = {key: 0.0 for key in EVENT_MEAN_METRICS}
        self.observations = {key: 0 for key in EVENT_MEAN_METRICS}

    def add(self, row: dict[str, Any]) -> None:
        self.n_events += 1
        for key in EVENT_COUNT_METRICS:
            self.counts[key] += int(row.get(key, 0))
        for key in EVENT_MEAN_METRICS:
            value = row.get(key)
            if value is not None:
                self.sums[key] += float(value)
                self.observations[key] += 1

    def summary(self) -> dict[str, Any]:
        out = {"n_events": self.n_events, **self.counts}
        for key in EVENT_MEAN_METRICS:
            n = self.observations[key]
            out[key] = finite_or_none(self.sums[key] / n) if n else None
        out["track_efficiency_global"] = (
            self.counts["n_matched_tracks"] / self.counts["n_true_tracks"]
            if self.counts["n_true_tracks"] else None
        )
        out["track_purity_global"] = (
            self.counts["n_matched_tracks"] / self.counts["n_pred_tracks"]
            if self.counts["n_pred_tracks"] else None
        )
        return out


def track_momentum_by_label(
    truth_labels: np.ndarray,
    reg_target: np.ndarray,
    background_label: int = -1,
    momentum_scale: float = 0.001,
) -> dict[int, dict[str, float]]:
    labels = np.asarray(truth_labels).reshape(-1)
    reg = np.asarray(reg_target)
    if reg.ndim != 2 or reg.shape[0] != labels.shape[0] or reg.shape[1] < 3:
        return {}
    out: dict[int, dict[str, float]] = {}
    for label in sorted(np.unique(labels).tolist()):
        if int(label) == int(background_label):
            continue
        rows = reg[labels == label, :3].astype(np.float64) * float(momentum_scale)
        rows = rows[np.isfinite(rows).all(axis=1)]
        if rows.size == 0:
            continue
        vec = rows.mean(axis=0)
        pt = float(np.hypot(vec[0], vec[1]))
        p = float(np.sqrt(np.dot(vec, vec)))
        out[int(label)] = {"p_gev": p, "pt_gev": pt}
    return out


# Opt-in offline evaluation. The legacy functions above and their consumers
# (including training/checkpoint selection) intentionally retain their contract.
MC_RECONSTRUCTION_VERSION = "mc_reconstruction_v1"


class ClusterProvenance(IntEnum):
    """Cluster composition, independent of whether its MC track is eligible."""

    UNKNOWN = 0
    MC_ONLY = 1
    OVERLAY_ONLY = 2
    MIXED = 3  # verified MC and measured-overlay members
    MC_PRESENT_UNRESOLVED = 4  # some MC; remaining member origins unresolved


def _point_array(value, name, size=None, *, integer=False, boolean=False):
    array = np.asarray(value)
    if array.ndim != 1 or (size is not None and len(array) != size):
        raise ValueError(f"{name} must be a one-dimensional aligned point array")
    if integer and array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer labels")
    if boolean and not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only booleans or 0/1")
    return array.astype(bool) if boolean else array


def cluster_provenance_from_hdf5(*, background_only, ownership=None,
                                 origin_noise=None, purity=None):
    """Conservatively interpret existing cvt-clusters-h5-v1 scalar sidecars.

    Only background_only (or its checked ownership=0 equivalent) verifies pure
    overlay. origin_noise=False establishes some MC, not MC-only composition.
    A strict-primary purity of exactly one verifies MC-only composition. The
    persisted scalars do not establish which other clusters are MC/overlay mixed.
    No truth labels are created, removed, or modified here.
    """
    bg = _point_array(background_only, "background_only", boolean=True)
    n = len(bg)
    result = np.full(n, ClusterProvenance.UNKNOWN, dtype=np.int8)
    if origin_noise is not None:
        noise = _point_array(origin_noise, "origin_noise", n, boolean=True)
        if np.any(bg & ~noise):
            raise ValueError("background_only contradicts MC origin evidence")
        result[~noise] = ClusterProvenance.MC_PRESENT_UNRESOLVED
    if ownership is not None:
        owner = _point_array(ownership, "ownership", n, integer=True)
        if not np.isin(owner, [0, 1, 2, 3]).all() or np.any((owner == 0) != bg):
            raise ValueError("ownership contradicts background_only or its schema")
        mc_evidence = np.isin(owner, [1, 2])
        if origin_noise is not None and np.any(mc_evidence & noise):
            raise ValueError("primary ownership contradicts origin_noise")
        result[mc_evidence] = ClusterProvenance.MC_PRESENT_UNRESOLVED
        if purity is not None:
            fraction = _point_array(purity, "purity", n)
            if not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
                raise ValueError("purity must be finite and in [0, 1]")
            if np.any((owner == 1) & (fraction <= 0.5)) or np.any(bg & (fraction != 0)):
                raise ValueError("purity contradicts ownership")
            # Exact equality: rounding a nearly-pure cluster up would overclaim.
            result[(owner == 1) & (fraction == 1)] = ClusterProvenance.MC_ONLY
    elif purity is not None:
        raise ValueError("purity requires ownership evidence")
    result[bg] = ClusterProvenance.OVERLAY_ONLY
    return result


_MC_COUNT_NAMES = (
    "n_mc_dice_matches", "n_mc_correct_points", "n_mc_associated_pred_points",
    "n_mc_other_track_points", "n_mc_nontarget_points",
    "n_split_true_tracks", "n_merged_pred_tracks", "n_non_target_only_pred_tracks",
    "n_mc_overlap_unassociated_pred_tracks", "n_background_assigned_points",
    "n_signal_unassigned_points", "n_truth_tracks_below_policy_min_clusters",
    "n_events_with_provenance",
) + tuple(f"n_{prefix}{kind}_points" for prefix in ("", "mc_associated_", "mc_nontarget_")
          for kind in ("mc_only", "overlay_only", "mixed", "mc_present_unresolved", "unknown"))


def _mc_ratios(counts):
    c = counts
    associated = c["n_mc_associated_pred_points"]
    overlay = c["n_mc_associated_overlay_only_points"]
    unknown = c["n_mc_associated_unknown_points"]
    ratios = {
        "mc_track_dice": _safe_div(c["mc_track_dice_sum"], c["n_true_tracks"]),
        "mc_hit_completeness": _safe_div(c["n_mc_correct_points"], c["n_signal_points"]),
        "mc_hit_purity": _safe_div(c["n_mc_correct_points"], associated),
        "mc_hit_impurity": _safe_div(associated - c["n_mc_correct_points"], associated),
        "mc_nontarget_contamination": _safe_div(c["n_mc_nontarget_points"], associated),
        "mc_other_track_contamination": _safe_div(c["n_mc_other_track_points"], associated),
        # These two metrics concern VERIFIED PURE overlay clusters, not all
        # non-target labels or an estimated fraction of mixed-cluster members.
        "mc_bg_contamination": None if unknown else _safe_div(overlay, associated),
        "bg_leakage_to_mc": (None if c["n_unknown_points"] else
                             _safe_div(overlay, c["n_overlay_only_points"])),
        "mc_bg_contamination_lower_bound": _safe_div(overlay, associated),
        "mc_provenance_coverage": _safe_div(associated - unknown, associated),
        "provenance_coverage": _safe_div(c["n_points"] - c["n_unknown_points"], c["n_points"]),
        "split_rate": _safe_div(c["n_split_true_tracks"], c["n_true_tracks"]),
        "merge_rate": _safe_div(c["n_merged_pred_tracks"], c["n_pred_tracks"]),
        "unmatched_predicted_fraction": _safe_div(c["n_pred_tracks"] - c["n_matched_tracks"], c["n_pred_tracks"]),
        "miss_rate": _safe_div(c["n_true_tracks"] - c["n_matched_tracks"], c["n_true_tracks"]),
        "non_target_only_predicted_fraction": _safe_div(c["n_non_target_only_pred_tracks"], c["n_pred_tracks"]),
        "mc_overlap_unassociated_predicted_fraction": _safe_div(c["n_mc_overlap_unassociated_pred_tracks"], c["n_pred_tracks"]),
        "background_contamination": _safe_div(c["n_background_assigned_points"],
            c["n_signal_points"] - c["n_signal_unassigned_points"] + c["n_background_assigned_points"]),
        "background_rejection": _safe_div(c["n_background_points"] - c["n_background_assigned_points"], c["n_background_points"]),
        "signal_loss_to_background": _safe_div(c["n_signal_unassigned_points"], c["n_signal_points"]),
    }
    for kind in ("mc_only", "overlay_only", "mixed", "mc_present_unresolved", "unknown"):
        ratios[f"mc_nontarget_{kind}_fraction"] = _safe_div(c[f"n_mc_nontarget_{kind}_points"], associated)
        ratios[f"mc_associated_{kind}_fraction"] = _safe_div(c[f"n_mc_associated_{kind}_points"], associated)
    return ratios


def event_mc_reconstruction_metrics(
    truth_labels, pred_labels, valid_mask=None, pred_signal_mask=None,
    config: MatchConfig | None = None, *, provenance=None, point_ids=None,
):
    """Add Phase-1 metrics to an unchanged legacy event result.

    Input is one complete event of FINAL disjoint prediction labels. Truth is
    the existing eligible-track label array, never reselected in this function.
    point_ids are unique original cluster indices, used only for reproducible
    assignment ties. See docs/evaluation/mc_reconstruction_v1.tex for the contract.
    """
    config = config or MatchConfig()
    truth = _point_array(truth_labels, "truth_labels", integer=True)
    pred = _point_array(pred_labels, "pred_labels", len(truth), integer=True)
    valid = (np.ones(len(truth), dtype=bool) if valid_mask is None else
             _point_array(valid_mask, "valid_mask", len(truth), boolean=True))
    selected = (pred >= 0 if pred_signal_mask is None else
                _point_array(pred_signal_mask, "pred_signal_mask", len(truth), boolean=True) & (pred >= 0))
    if np.any(valid & (truth < 0) & (truth != config.background_label)):
        raise ValueError("Valid truth has an unknown negative label; mask padding explicitly")
    points = (np.arange(len(truth)) if point_ids is None else
              _point_array(point_ids, "point_ids", len(truth), integer=True))
    if len(np.unique(points[valid])) != int(valid.sum()):
        raise ValueError("point_ids must be unique within an event")
    origins = (np.full(len(truth), ClusterProvenance.UNKNOWN, dtype=np.int8) if provenance is None else
               _point_array(provenance, "provenance", len(truth), integer=True))
    if not np.isin(origins[valid], list(ClusterProvenance)).all():
        raise ValueError("Unknown provenance code")
    signal = valid & (truth != config.background_label)
    if np.any(signal & (origins == ClusterProvenance.OVERLAY_ONLY)):
        raise ValueError("Eligible MC truth cannot be verified pure overlay")
    selected = selected & valid
    # Canonical order by the smallest ORIGINAL cluster index in each disjoint
    # set. Independent of numerical track IDs and of input serialization when
    # stable point_ids are supplied. No epsilon perturbation of Dice weights.
    true_ids = sorted(np.unique(truth[signal]), key=lambda t: int(points[signal & (truth == t)].min()))
    pred_ids = sorted(np.unique(pred[selected]), key=lambda p: int(points[selected & (pred == p)].min()))
    tsets = [signal & (truth == t) for t in true_ids]
    psets = [selected & (pred == p) for p in pred_ids]
    tsizes = np.array([int(t.sum()) for t in tsets], dtype=np.int64)
    psizes = np.array([int(p.sum()) for p in psets], dtype=np.int64)
    intersections = np.zeros((len(psets), len(tsets)), dtype=np.int64)
    for j, p in enumerate(psets):
        for i, t in enumerate(tsets):
            intersections[j, i] = np.count_nonzero(p & t)
    dice = 2.0 * intersections / (psizes[:, None] + tsizes[None, :])
    matches, associated = [], np.zeros(len(truth), dtype=bool)
    if dice.size:
        # Independent non-target predictions must not even influence tie order.
        active = np.flatnonzero(intersections.sum(axis=1) > 0)
        js, its = linear_sum_assignment(-dice[active])
        for local_j, i in zip(js, its):
            j = active[local_j]
            if intersections[j, i] == 0:
                continue  # zero-overlap solver pairs are NOT MC associations
            associated |= psets[j]
            matches.append({"true_id": int(true_ids[i]), "pred_id": int(pred_ids[j]),
                "intersection": int(intersections[j, i]), "true_size": int(tsizes[i]),
                "pred_size": int(psizes[j]), "dice": float(dice[j, i])})
    # Independent eligible-first IoU matching, with its established thresholds.
    row = event_track_metrics(truth, pred, valid, selected, config)
    row.update({
        "mc_reconstruction_version": MC_RECONSTRUCTION_VERSION,
        "legacy_match_config": dict(vars(config)),
        "dice_matches": matches,
        "mc_track_dice_sum": float(sum(m["dice"] for m in matches)),
        "n_mc_dice_matches": len(matches),
        "n_mc_correct_points": sum(m["intersection"] for m in matches),
        "n_mc_associated_pred_points": int(associated.sum()),
        "n_mc_other_track_points": int((associated & signal).sum()) - sum(m["intersection"] for m in matches),
        "n_mc_nontarget_points": int((associated & valid & ~signal).sum()),
        "n_split_true_tracks": int(((intersections > 0).sum(axis=0) > 1).sum()),
        "n_merged_pred_tracks": int(((intersections > 0).sum(axis=1) > 1).sum()),
        "n_non_target_only_pred_tracks": int((intersections.sum(axis=1) == 0).sum()),
        "n_mc_overlap_unassociated_pred_tracks": int((intersections.sum(axis=1) > 0).sum()) - len(matches),
        "n_background_assigned_points": int((valid & ~signal & selected).sum()),
        "n_signal_unassigned_points": int((signal & ~selected).sum()),
        # Audit only: never silently impose a new eligibility cut on old labels.
        "n_truth_tracks_below_policy_min_clusters": int((tsizes < 4).sum()),
        "n_events_with_provenance": int(provenance is not None),
    })
    for prefix, mask in (("", valid), ("mc_associated_", associated),
                         ("mc_nontarget_", associated & ~signal)):
        for kind in ClusterProvenance:
            row[f"n_{prefix}{kind.name.lower()}_points"] = int((mask & (origins == kind)).sum())
    row["mc_event_metrics"] = _mc_ratios(row)
    return row


class MCReconstructionAccumulator:
    """Pool sufficient counts, not batch means; retain all legacy summaries.

    add() accepts complete-event results. merge() accepts another accumulator
    with the same profile/accepted-match configuration (e.g. one per batch).
    """

    def __init__(self):
        self.legacy = EventMetricAccumulator()
        self.counts = {k: 0 for k in _MC_COUNT_NAMES}
        self.dice_sum = 0.0
        self.accepted_sums = {k: 0.0 for k in ("iou", "purity", "efficiency")}
        self.macro_sums, self.macro_n = {}, {}
        self.match_config = None

    def _check_config(self, config):
        if self.match_config is not None and self.match_config != config:
            raise ValueError("Cannot combine different accepted-match configurations")
        self.match_config = dict(config)

    def add(self, row):
        if row.get("mc_reconstruction_version") != MC_RECONSTRUCTION_VERSION:
            raise ValueError("Expected a versioned MC reconstruction event row")
        self._check_config(row["legacy_match_config"])
        self.legacy.add(row)
        for k in self.counts:
            self.counts[k] += int(row[k])
        self.dice_sum += row["mc_track_dice_sum"]
        for k in self.accepted_sums:
            self.accepted_sums[k] += sum(m[k] for m in row["matches"])
        for k, value in row["mc_event_metrics"].items():
            if value is not None:
                self.macro_sums[k] = self.macro_sums.get(k, 0.0) + value
                self.macro_n[k] = self.macro_n.get(k, 0) + 1

    def merge(self, other):
        if not isinstance(other, MCReconstructionAccumulator):
            raise TypeError("Expected MCReconstructionAccumulator")
        if other is self:
            raise ValueError("Cannot merge an accumulator into itself")
        if other.match_config is not None:
            self._check_config(other.match_config)
        self.legacy.n_events += other.legacy.n_events
        for dest, source in ((self.legacy.counts, other.legacy.counts),
                             (self.legacy.sums, other.legacy.sums),
                             (self.legacy.observations, other.legacy.observations),
                             (self.counts, other.counts), (self.accepted_sums, other.accepted_sums),
                             (self.macro_sums, other.macro_sums), (self.macro_n, other.macro_n)):
            for k, v in source.items():
                dest[k] = dest.get(k, 0) + v
        self.dice_sum += other.dice_sum

    def summary(self):
        result = self.legacy.summary()
        # Explicit aliases: the historical unsuffixed values remain unchanged.
        result.update({k + "_event_macro": result[k] for k in EVENT_MEAN_METRICS})
        result.update(self.counts)
        result["mc_track_dice_sum"] = self.dice_sum
        for k, v in _mc_ratios(result).items():
            result["mc_track_dice_mean" if k == "mc_track_dice" else k + "_global"] = v
            result[k + "_event_macro"] = _safe_div(self.macro_sums.get(k, 0), self.macro_n.get(k, 0))
        for k, v in self.accepted_sums.items():
            result[f"matched_{k}_mean_global"] = _safe_div(v, result["n_matched_tracks"])
        result.update({"mc_reconstruction_version": MC_RECONSTRUCTION_VERSION,
                       "legacy_match_config": self.match_config,
                       "mc_event_macro_observations": dict(self.macro_n)})
        return result


def summarize_mc_reconstruction_metrics(rows):
    accumulator = MCReconstructionAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.summary()
