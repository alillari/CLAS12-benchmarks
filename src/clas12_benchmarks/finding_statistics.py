"""Event bootstrap of MC sufficient counts, conditional on the trained weights."""
import numpy as np

from .finding_evaluation import MC_HEADLINES, require_same_contract


def mc_intervals(rows, baseline=None, *, seed=1337, replicates=1000):
    if not rows or replicates < 1:
        raise ValueError("Nonempty events and positive bootstrap replicates required")
    if baseline is not None and len(rows) != len(baseline):
        raise ValueError("Paired bootstrap populations differ")
    columns = ("mc_track_dice_sum", "n_true_tracks", "n_mc_correct_points", "n_signal_points",
               "n_mc_associated_pred_points", "n_mc_associated_overlay_only_points",
               "n_mc_associated_unknown_points", "n_overlay_only_points", "n_unknown_points")
    def values(records):
        for row in records:
            require_same_contract(rows[0], row)
            if row.get("mc_reconstruction_version") != "mc_reconstruction_v1":
                raise ValueError("Mixed or missing MC metric profile in bootstrap rows")
        return np.array([[row[k] for k in columns] for row in records], dtype=np.float64)
    a = values(rows)
    b = values(baseline) if baseline is not None else None
    if baseline is not None:
        for x, y in zip(rows, baseline, strict=True):
            if "identity" not in x or "identity" not in y or x["identity"] != y["identity"]:
                raise ValueError("Paired evaluation identities differ")
    def divide(x, y):
        return np.divide(x, y, out=np.full_like(x, np.nan), where=y != 0)
    def summarize(s):
        return np.column_stack((divide(s[:, 0], s[:, 1]), divide(s[:, 2], s[:, 3]),
            divide(s[:, 2], s[:, 4]), divide(2*s[:, 2], s[:, 3]+s[:, 4]),
            np.where(s[:, 6] == 0, divide(s[:, 5], s[:, 4]), np.nan),
            np.where(s[:, 8] == 0, divide(s[:, 5], s[:, 7]), np.nan)))
    samples, rng = [], np.random.default_rng(seed)
    for start in range(0, replicates, 32):
        indices = rng.integers(len(rows), size=(min(32, replicates-start), len(rows)))
        result = summarize(a[indices].sum(axis=1))
        if b is not None:
            result -= summarize(b[indices].sum(axis=1))
        samples.extend(result.tolist())
    samples = np.asarray(samples)
    intervals, defined = {}, {}
    for i, key in enumerate(MC_HEADLINES):
        finite = samples[:, i][np.isfinite(samples[:, i])]
        intervals[key] = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None
        defined[key] = len(finite)
    return {"intervals_95": intervals, "defined_resamples": defined,
            "unit": "paired event" if baseline is not None else "event",
            "replicates": replicates, "seed": seed}
