"""Pooled cluster operating points and event-level uncertainty."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from .fm_reference.noise_metrics import (
    at_threshold, calibrate_thresholds, retention_threshold, score_summary,
)

RETENTION_TARGETS = {"retain_95pct": .95, "retain_98pct": .98, "retain_99pct": .99,
                     "retain_99p5pct": .995}
REJECTION_TARGETS = {"reject_90pct": .90, "reject_95pct": .95, "reject_98pct": .98,
                     "reject_99pct": .99}


def calibration(labels, scores, coatjava):
    points = {"default": {"threshold": .5}, **calibrate_thresholds(labels, scores, coatjava)}
    for name, target in RETENTION_TARGETS.items():
        points[name] = dict(threshold=retention_threshold(labels, scores, target),
                            target_signal_retention=target)
    noise = np.sort(scores[labels == 1])
    if not len(noise):
        raise ValueError("Cannot calibrate without noise clusters")
    for name, target in REJECTION_TARGETS.items():
        index = len(noise) - int(np.ceil(len(noise) * target))
        below = np.nextafter(noise[index], scores.dtype.type(-np.inf)).item()
        points[name] = dict(threshold=below, target_noise_rejection=target)
    signal = np.sort(scores[labels == 0])
    thresholds = np.unique(np.r_[-np.finfo(scores.dtype).eps, scores.astype(np.float64)])
    tn = np.searchsorted(signal, thresholds, side="right")
    tp = len(noise) - np.searchsorted(noise, thresholds, side="right")
    objective = tn * len(noise) + tp * len(signal)
    points["max_balanced_accuracy"] = {"threshold": float(thresholds[np.argmax(objective)])}
    return points


def operating_points(labels, scores, points):
    return {name: {**point, **at_threshold(labels, scores, point["threshold"])}
            for name, point in points.items()}


def grouped_results(labels, scores, coatjava, points, offsets, detector, layer):
    occupancy = np.repeat(np.diff(offsets), np.diff(offsets))
    groups = {}
    for low, high in [(0, 100), (100, 150), (150, 200), (200, 250), (250, np.inf)]:
        upper = str(high - 1) if np.isfinite(high) else "inf"
        groups[f"occupancy_{low}_{upper}"] = (occupancy >= low) & (occupancy < high)
    for code, name in [(1, "BMT"), (2, "BST")]:
        groups[name] = detector == code
        for number in sorted(set(layer[groups[name]].tolist())):
            groups[f"{name}_layer_{number}"] = groups[name] & (layer == number)
    output = {}
    selected = ("default", "retain_95pct", "retain_98pct", "retain_99pct", "retain_99p5pct")
    for name, mask in groups.items():
        if mask.any():
            summary = score_summary(labels[mask], scores[mask], coatjava[mask])
            summary["operating_points"] = operating_points(
                labels[mask], scores[mask], {key: points[key] for key in selected})
            output[name] = summary
    return output


def event_confusions(labels, prediction, offsets):
    codes = 2 * labels + prediction
    cumulative = np.vstack([np.r_[0, np.cumsum(codes == value)] for value in range(4)])
    return np.diff(cumulative[:, offsets], axis=1).T


def bootstrap_rates(counts_by_model, draws=400, seed=73):
    """Use common row draws for paired comparisons of fixed operating points."""
    if draws < 1 or not counts_by_model:
        raise ValueError("Require positive bootstrap draws and at least one model")
    names = list(counts_by_model)
    values = np.stack([counts_by_model[name] for name in names])
    samples = np.empty((draws, len(names), 2))
    count = values.shape[1]
    rng = np.random.default_rng(seed)
    for index in range(draws):
        tn, fp, fn, tp = values[:, rng.integers(count, size=count)].sum(axis=1).T
        denominators = np.stack((tn + fp, fn + tp), axis=-1)
        rates = np.stack((tn, tp), axis=-1)
        samples[index] = np.divide(rates, denominators,
                                   out=np.full(rates.shape, np.nan), where=denominators != 0)

    def intervals(rates):
        return {metric: np.quantile(column[np.isfinite(column)], [.025, .975]).tolist()
                if np.isfinite(column).any() else None
                for metric, column in zip(("signal_retention", "noise_rejection"), rates.T, strict=True)}

    return {"draws": draws, "seed": seed, "unit": "event",
            "absolute_95pct": {name: intervals(samples[:, i]) for i, name in enumerate(names)},
            "paired_difference_95pct": {f"{name}_minus_{baseline}": intervals(samples[:, i] - samples[:, j])
                                        for i, name in enumerate(names) for j, baseline in enumerate(names[:i])}}


def exact_curves(labels, scores):
    signal_score = 1.0 - scores.astype(np.float64)
    t_noise, t_signal, roc_threshold = roc_curve(labels == 0, signal_score, drop_intermediate=False)
    precision, recall, pr_threshold = precision_recall_curve(labels == 0, signal_score)
    return dict(false_signal_rate=t_noise, signal_retention=t_signal, noise_rejection=1.0 - t_noise,
                roc_signal_score_threshold=roc_threshold, signal_precision=precision,
                signal_recall=recall, pr_signal_score_threshold=pr_threshold,
                noise_average_precision=average_precision_score(labels == 1, scores))
