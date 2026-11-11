"""Binary noise metrics and validation-only filtering thresholds (0=signal)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_DEFINITIONS = {
    "unit": "cluster; counts pooled over all evaluated events, padding excluded",
    "labels": {"0": "signal", "1": "noise"},
    "accuracy": "correctly classified clusters / all clusters",
    "signal_precision": "true signal retained / all clusters predicted signal",
    "signal_recall": "true signal retained / all true signal clusters",
    "noise_precision": "true noise rejected / all clusters predicted noise",
    "noise_recall": "true noise rejected / all true noise clusters",
    "balanced_accuracy": "mean of signal recall and noise recall",
    "macro_precision": "mean of signal precision and noise precision",
    "macro_recall": "mean of signal recall and noise recall",
    "undefined": "null when a denominator is zero; macro metrics require both classes",
}


def inverse_frequency_weights(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (2,) or np.any(counts <= 0):
        raise ValueError("Balanced noise training requires both signal and noise clusters")
    return counts.sum() / (2.0 * counts)


def training_class_counts(dataset):
    """Count the accepted training events, without consulting validation/test."""
    if dataset.chunk_training:
        raise ValueError("Noise class balancing currently requires whole-event samples")
    counts = np.zeros(2, dtype=np.int64)
    for index in dataset.idxlist:
        labels = np.asarray(dataset.memmap_noise_target[int(index)])
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Expected binary noise labels in the training cohort")
        counts += np.bincount(labels.astype(np.int64), minlength=2)
    return counts


def confusion_metrics(matrix):
    matrix = np.asarray(matrix)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or np.any(matrix != np.floor(matrix))):
        raise ValueError("Expected a nonnegative integer 2x2 confusion matrix")
    matrix = matrix.astype(np.int64)
    tn, fp, fn, tp = matrix.ravel()
    div = lambda x, y: float(x / y) if y else None
    mean = lambda a, b: None if a is None or b is None else (a + b) / 2
    signal_precision, signal_recall = div(tn, tn + fn), div(tn, tn + fp)
    noise_precision, noise_recall = div(tp, tp + fp), div(tp, tp + fn)
    return dict(confusion_true_rows_pred_columns=matrix.tolist(), n_clusters=int(matrix.sum()),
                noise_precision=noise_precision, noise_recall=noise_recall,
                signal_precision=signal_precision, signal_recall=signal_recall,
                signal_retention=signal_recall, noise_rejection=noise_recall,
                balanced_accuracy=mean(signal_recall, noise_recall),
                macro_precision=mean(signal_precision, noise_precision),
                macro_recall=mean(signal_recall, noise_recall),
                accuracy=div(tp + tn, matrix.sum()))


def at_threshold(labels, noise_score, threshold):
    prediction = np.asarray(noise_score) > threshold
    matrix = np.bincount(2 * np.asarray(labels, dtype=np.int64) + prediction, minlength=4).reshape(2, 2)
    return dict(threshold=float(threshold), **confusion_metrics(matrix))


def retention_threshold(labels, noise_score, retention):
    if not 0 < retention <= 1:
        raise ValueError("Signal retention must lie in (0, 1]")
    signal = np.asarray(noise_score)[np.asarray(labels) == 0]
    if not len(signal):
        raise ValueError("Cannot calibrate without signal clusters")
    # Reject strictly above threshold; ties are retained conservatively.
    return float(np.sort(signal)[max(0, int(np.ceil(retention * len(signal))) - 1)])


def score_summary(labels, noise_score, coatjava_noise):
    labels = np.asarray(labels, dtype=np.int64)
    score = np.asarray(noise_score)
    if not np.isfinite(score).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid noise evaluation scores or labels")
    counts = np.bincount(labels, minlength=2)
    both = bool(np.all(counts > 0))
    matrix = np.bincount(2 * labels + np.asarray(coatjava_noise, dtype=np.int64), minlength=4).reshape(2, 2)
    return dict(
        metric_definitions=METRIC_DEFINITIONS,
        signal_clusters=int(counts[0]), noise_clusters=int(counts[1]),
        all_noise_accuracy=float(counts[1] / counts.sum()),
        signal_roc_auc=float(roc_auc_score(labels == 0, 1. - score)) if both else None,
        signal_average_precision=float(average_precision_score(labels == 0, 1. - score)) if both else None,
        noise_score_quantiles={str(c): np.quantile(score[labels == c], [0., .1, .5, .9, 1.]).tolist()
                               if counts[c] else [] for c in (0, 1)},
        adapter=at_threshold(labels, score, .5), coatjava_unassigned=confusion_metrics(matrix),
    )


def calibrate_thresholds(labels, noise_score, coatjava_noise):
    baseline = score_summary(labels, noise_score, coatjava_noise)["coatjava_unassigned"]
    targets = {"retain_95pct": .95, "retain_99pct": .99,
               "match_coatjava_retention": baseline["signal_retention"]}
    return {name: dict(target_signal_retention=target,
                       **at_threshold(labels, noise_score, retention_threshold(labels, noise_score, target)))
            for name, target in targets.items() if target is not None and target > 0}
