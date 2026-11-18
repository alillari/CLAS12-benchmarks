"""Lock validation winners, then score independent strip test cohorts and paired CVT."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .strip_adapter import HYBRID, KINDS, StripAdapter, errors, load_data, metrics, predict, write_json


def load_stage(path):
    ranking = json.loads((path / "ranking.json").read_text())
    cohort = json.loads((path / "cohort.json").read_text())
    # Keep the original rankings readable, including archived FM reference runs.
    fits = [(row, path / row["arm"]) for row in ranking if row["model"] in KINDS]
    if not fits:
        raise ValueError(f"No standalone strip models in {path}")
    return fits, cohort


def select(fits, val):
    ranked = sorted(fits, key=lambda item: item[0]["best"]["p_mae_gev"])
    truth = val["truth"].cpu().numpy()
    predictions = [np.load(folder / "validation_predictions.npy") for _, folder in ranked]
    single = metrics(truth, predictions[0])
    assert abs(single["p_mae_gev"] - ranked[0][0]["best"]["p_mae_gev"]) < 1e-6
    candidates = []
    for size in (2, 3, 5):
        if size <= len(ranked):
            score = metrics(truth, np.mean(predictions[:size], axis=0))
            candidates.append((score["p_mae_gev"], size, score))
    candidates.append((single["p_mae_gev"], 1, single))
    _, selected_size, validation_score = min(candidates)
    return ranked, selected_size, single, validation_score, [
        dict(size=n, validation=score) for _, n, score in candidates]


def source_bootstrap(truth, adapter, coatjava, identity, n_reps=1000):
    """Keep tracks from the same production source together."""
    first = np.abs(errors(truth, adapter)[0])
    second = np.abs(errors(truth, coatjava)[0])
    _, index = np.unique(identity[:, 0], return_inverse=True)
    counts = np.bincount(index)
    sums = np.bincount(index, weights=first - second)
    if len(counts) < 2:
        raise ValueError("Paired source bootstrap needs multiple sources")
    rng = np.random.default_rng(47542)
    draws = []
    for _ in range(n_reps):
        sample = rng.integers(len(counts), size=len(counts))
        draws.append(sums[sample].sum() / counts[sample].sum())
    return dict(delta_p_mae_gev=float(np.mean(first - second)), sources=len(counts),
                replicates=n_reps, ci95_gev=np.quantile(draws, [.025, .975]).tolist())


def breakdown(test, truth, pred, cvt, paired):
    p = np.linalg.norm(truth, axis=1)
    theta = np.degrees(np.arctan2(np.linalg.norm(truth[:, :2], axis=1), truth[:, 2]))
    groups = defaultdict(dict)
    for key, values, edges in (
        ("momentum_GeV_c", p, [0, .5, 1, 1.5, 2, 3, 4, 6, np.inf]),
        ("theta_deg", theta, [0, 45, 60, 75, 90, 105, 120, 135, 180]),
    ):
        for lo, hi in zip(edges[:-1], edges[1:]):
            groups[key][f"{lo:g}–{hi:g}"] = (values >= lo) & (values < hi)
    for pid in np.unique(test["pid"]):
        groups["signed_PDG"][str(pid)] = test["pid"] == pid
    layers = (test["features"]["geometry"][:, 18::19] > 0).sum(1).cpu().numpy()
    for n in np.unique(layers):
        groups["distinct_strip_layers"][str(n)] = layers == n
    groups["published_minimum_coverage"] = {"meets": test["coverage"][:, 4] == 1,
                                            "below": test["coverage"][:, 4] == 0}
    return {
        name: {label: dict(adapter_all=metrics(truth[mask], pred[mask]),
                           adapter_paired=metrics(truth[mask & paired], pred[mask & paired]),
                           coatjava_paired=metrics(truth[mask & paired], cvt[mask & paired]))
               for label, mask in named_masks.items()}
        for name, named_masks in groups.items()
    }


def load_model(row, folder, manifest_sha, device):
    checkpoint = torch.load(folder / "best.pth", map_location=device, weights_only=True)
    assert checkpoint["manifest_sha256"] == manifest_sha
    assert checkpoint["step"] == row["best"]["step"]
    model = StripAdapter(*checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["state"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--final", type=Path, default=Path("/final"))
    parser.add_argument("--large", type=Path)
    parser.add_argument("--large-cache", type=Path)
    parser.add_argument("--train-cache", type=Path,
                        help="Training cohort cache when it is separate from --validation")
    parser.add_argument("--validation", type=Path, default=Path("/cache"))
    parser.add_argument("--test", type=Path, default=Path("/test"))
    parser.add_argument("--out", type=Path, default=Path("/results"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    finalists, final_cohort = load_stage(args.final)
    stages = [(args.final, "scale", finalists, final_cohort)]
    if args.screen:
        screening, cohort = load_stage(args.screen)
        stages.insert(0, (args.screen, "screen", screening, cohort))
    if args.large:
        larger, large_cohort = load_stage(args.large)
        stages.append((args.large, "large", larger, large_cohort))
    val = load_data(args.validation, "validation", final_cohort["validation_tracks"], False, device)
    manifest_sha = val["report"]["manifest_sha256"]
    assert final_cohort["val_prefix_sha256"] == hashlib.sha256(val["identity"].tobytes()).hexdigest()
    stage_cohorts = {path: (stage, cohort) for path, stage, _, cohort in stages}
    for _, cohort in stage_cohorts.values():
        assert cohort["val_prefix_sha256"] == final_cohort["val_prefix_sha256"]
        assert cohort["train"]["manifest_sha256"] == manifest_sha
    fits = [item for _, _, stage_fits, _ in stages for item in stage_fits]
    ranked, n, single, selected, candidates = select(fits, val)
    del val

    def result_name(row, folder):
        stage, _ = stage_cohorts[folder.parent]
        return f"{stage}/{row['arm']}"

    def identifier(item):
        row, folder = item
        stage, cohort = stage_cohorts[folder.parent]
        return dict(arm=row["arm"], name=result_name(row, folder),
                    validation=row["best"], parameters=row["parameters"],
                    stage=stage, train_tracks=cohort["train_tracks"],
                    checkpoint_sha256=hashlib.sha256((folder / "best.pth").read_bytes()).hexdigest())
    selection = {
        "objective": "minimum validation p MAE",
        "best_single": identifier(ranked[0]), "single_validation": single,
        "ensemble_members": [identifier(item) for item in ranked[:n]] if n > 1 else [],
        "selected_validation": selected, "validation_ensemble_candidates": candidates,
        "validation_tracks": final_cohort["validation_tracks"],
    }
    write_json(args.out / "selection.json", selection)
    needed = ranked[:n]
    if args.screen and screening[0] not in needed:
        needed.append(screening[0])
    need_raw = any(row["model"] == HYBRID for row, _ in needed)
    test = load_data(args.test, "test", None, need_raw, device)
    assert test["report"]["manifest_sha256"] == manifest_sha
    sources = set(test["identity"][:, 0])
    caches = [(args.validation, "validation"), (args.train_cache or args.validation, "train")]
    if args.screen and args.train_cache:
        caches.append((args.validation, "train"))
    if args.large_cache:
        caches.append((args.large_cache, "train"))
    for root, split in caches:
        identities = np.load(root / f"{split}_identity.npy", mmap_mode="r")
        assert sources.isdisjoint(identities[:, 0])
    predictions = {}
    for row, folder in needed:
        model, checkpoint = load_model(row, folder, manifest_sha, device)
        predictions[result_name(row, folder)] = predict(
            model, test, checkpoint["loss"], checkpoint["stats"], checkpoint["amp"])
    if n > 1:
        predictions["validation_selected_ensemble"] = np.mean(
            [predictions[result_name(*item)] for item in ranked[:n]], axis=0)

    truth = test["truth"].cpu().numpy()
    cvt = test["coatjava"]
    paired = np.isfinite(cvt).all(1)
    conventional = metrics(truth[paired], cvt[paired])
    summary = {}
    for name, pred in predictions.items():
        summary[name] = dict(all_tracks=metrics(truth, pred),
                             paired_with_coatjava=metrics(truth[paired], pred[paired]),
                             paired_p_mae_difference=source_bootstrap(
                                 truth[paired], pred[paired], cvt[paired], test["identity"][paired]))
    best_single_name = result_name(*ranked[0])
    best_name = "validation_selected_ensemble" if n > 1 else best_single_name
    report = dict(test_cohort=test["report"], test_results=summary, best_test_entry=best_name,
                  best_single_entry=best_single_name,
                  coatjava_paired=conventional, input_scope="truth-grouped direct-primary strips",
                  differential=breakdown(test, truth, predictions[best_single_name], cvt, paired))
    write_json(args.out / "test_metrics.json", report)
    np.savez_compressed(args.out / "test_predictions.npz", identity=test["identity"],
                        truth=truth, coatjava=cvt, pid=test["pid"], coverage=test["coverage"],
                        **predictions)
    print(json.dumps({k: v for k, v in report.items() if k != "differential"}), flush=True)


if __name__ == "__main__":
    main()
