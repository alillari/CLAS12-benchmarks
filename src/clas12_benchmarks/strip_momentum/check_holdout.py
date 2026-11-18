"""Recompute held-out metrics and quantify the native momentum outlier sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate_strips import source_bootstrap
from .strip_adapter import errors, metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("models", type=Path)
    args = parser.parse_args()
    report = json.loads((args.evidence / "test_metrics.json").read_text())
    selection = json.loads((args.evidence / "selection.json").read_text())
    for name, entry in (("best_single.pth", selection["best_single"]),
                        ("ensemble_second.pth", selection["ensemble_members"][1])):
        actual = hashlib.sha256((args.models / name).read_bytes()).hexdigest()
        assert actual == entry["checkpoint_sha256"], name
    with np.load(args.evidence / "test_predictions.npz") as data:
        truth, cvt = data["truth"], data["coatjava"]
        keys = data["identity"]
        counts = len(keys)
        assert len(np.unique(keys, axis=0)) == counts
        assert hashlib.sha256(keys.tobytes()).hexdigest() == report["test_cohort"]["identity_sha256"]
        valid = np.isfinite(cvt).all(1)
        cvt_norm = np.linalg.norm(cvt, axis=1)
        p = np.linalg.norm(truth, axis=1)
        names = list(report["test_results"])
        for name in names:
            adapter = data[name]
            for label, mask in (("all_tracks", np.ones(counts, bool)), ("paired_with_coatjava", valid)):
                actual = metrics(truth[mask], adapter[mask])
                for key, value in report["test_results"][name][label].items():
                    np.testing.assert_allclose(actual[key], value, rtol=1e-6, atol=2e-7)
        for key, expected in report["coatjava_paired"].items():
            np.testing.assert_allclose(metrics(truth[valid], cvt[valid])[key], expected, rtol=1e-6, atol=2e-7)

        cvt_abs_error = np.abs(errors(truth[valid], cvt[valid])[0])
        pair_indices = np.flatnonzero(valid)
        sorted_errors = np.argsort(cvt_abs_error)[::-1]
        worst_idx = pair_indices[sorted_errors[:10]]
        largest = [dict(identity=keys[i].tolist(), truth_p_gev=float(p[i]), cvt_p_gev=float(cvt_norm[i]))
                   for i in worst_idx]
        total_error = cvt_abs_error.sum()
        sensitivity = dict(
            finite_pairs=int(valid.sum()),
            truth_max_p_gev=float(p.max()),
            largest_native_errors=largest,
            largest_one_share_of_absolute_error=float(cvt_abs_error[sorted_errors[0]] / total_error),
            thresholds={},
        )
        best = data[report["best_single_entry"]]
        for cap in (10, 20, 50, 100):
            keep = valid & (cvt_norm <= cap)
            after = dict(tracks=int(keep.sum()), removed=int(valid.sum() - keep.sum()),
                         coatjava=metrics(truth[keep], cvt[keep]),
                         adapter=metrics(truth[keep], best[keep]),
                         delta_mae=source_bootstrap(truth[keep], best[keep], cvt[keep], keys[keep]))
            if cap == 20:
                edges = (0, .5, 1, 1.5, 2, 3, 4, 6, np.inf)
                after["momentum_bins"] = {}
                for lo, hi in zip(edges[:-1], edges[1:]):
                    mask = keep & (p >= lo) & (p < hi)
                    after["momentum_bins"][f"{lo:g}–{hi:g}"] = dict(
                        adapter_paired=metrics(truth[mask], best[mask]),
                        coatjava_paired=metrics(truth[mask], cvt[mask]))
            sensitivity["thresholds"][str(cap)] = after
        purity_path = args.evidence / "paired_cvt_purity.npy"
        if purity_path.is_file():
            purity = np.load(purity_path)
            assert np.array_equal(np.isfinite(purity), valid)
            sensitivity["minimum_cvt_purity"] = {}
            for threshold in (.8, .9, .99):
                keep = valid & (purity >= threshold) & (cvt_norm <= 20)
                sensitivity["minimum_cvt_purity"][str(threshold)] = dict(
                    tracks=int(keep.sum()), coatjava=metrics(truth[keep], cvt[keep]),
                    adapter=metrics(truth[keep], best[keep]),
                    delta_mae=source_bootstrap(truth[keep], best[keep], cvt[keep], keys[keep]),
                )
    write_json(args.evidence / "coatjava_tail_sensitivity.json", sensitivity)
    print(f"Verified {counts:,} track identities, {int(valid.sum()):,} CVT matches, "
          f"{len(names)} prediction arrays and both checkpoint hashes")


if __name__ == "__main__":
    main()
