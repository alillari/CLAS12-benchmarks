"""Bounded HDF5 CLI, train/reload/evaluate/audit check on synthetic or real events."""
import argparse
import csv
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.momentum_manifest import TrackManifest
from clas12_benchmarks.upstream import ROOT


def check_resume(method, cohort, recipe, metrics, baseline, output):
    """Interrupt after checkpoint 2 with the original total budget unchanged."""
    import torch
    from clas12_benchmarks.tracking import run
    class Interrupted(RuntimeError):
        pass
    original, calls = torch.optim.AdamW.step, 0
    def interrupted_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise Interrupted("planned interruption before update 3")
        return original(optimizer, *args, **kwargs)
    args = SimpleNamespace(method=method, cohort=str(cohort), config=str(recipe), metrics=str(metrics),
                           output=str(output), device="cpu", threads=2, smoke=False, resume=False)
    with patch.object(torch.optim.AdamW, "step", interrupted_step):
        try:
            run(args)
        except Interrupted:
            pass
        else:
            raise ValueError("Planned interruption did not execute")
    args.resume = True
    run(args)
    for stage in json.loads((baseline / "selected_checkpoints.json").read_text()):
        for suffix in ("last", "best"):
            name = f"{stage['stage']}_{suffix}.pt"
            a = torch.load(baseline / name, map_location="cpu", weights_only=False)
            b = torch.load(output / name, map_location="cpu", weights_only=False)
            if a["step"] != b["step"]:
                raise ValueError("Resume update count changed")
            for key, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
    for filename in ("summary.json", "frozen_selection.json"):
        if json.loads((baseline / filename).read_text()) != json.loads((output / filename).read_text()):
            raise ValueError(f"Resume changed {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tracking.json")
    parser.add_argument("--check-recovery", action="store_true", help="CPU interruption/resume and fresh holdout checks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.data
    if root is None:
        sys.path.insert(0, str(ROOT / "tests"))
        from hdf5_fixture import make_publication
        root = make_publication(args.output / "fixture")
    counts = (16, 8, 8) if args.data else (3, 2, 2)
    def run(name, argv):
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run([sys.executable, *map(str, argv)], check=True, stdout=log, stderr=subprocess.STDOUT)
    cohort = args.output / "cohort/cohort.json"
    run("prepare", ["-m", "clas12_benchmarks.prepare_cohort", root, cohort.parent, "--reader", "hdf5",
                    "--train", counts[0], "--val", counts[1], "--test", counts[2], "--seed", 1337])
    run("inspect", ["-m", "clas12_benchmarks", "inspect", cohort])
    run("export", ["-m", "clas12_benchmarks", "export-eggnet", cohort, "--output", args.output / "eggnet-export"])
    pack = args.output / "tracks.npz"
    run("momentum", ["-m", "clas12_benchmarks", "prepare-momentum-hdf5", "--cohort", cohort,
                     "--output", pack, "--min-clusters", 12 if args.data else 1])
    run("inspect-momentum", ["-m", "clas12_benchmarks", "inspect-momentum", pack])
    tracks = TrackManifest(pack)
    rows = tracks.rows("evaluation")
    if not len(rows):
        raise ValueError("Smoke population has no momentum evaluation tracks")
    predictions = args.output / "perfect-predictions.csv"
    with predictions.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*tracks.identity_fields, "schema", "momentum_unit", "manifest_sha256",
                         "pred_px_gev", "pred_py_gev", "pred_pz_gev"])
        for row in reversed(rows):
            writer.writerow([*tracks.identity(row), "clas12_momentum_predictions_v2", "GeV",
                             tracks.sha256, *tracks.truth[row]])
    run("score", ["-m", "clas12_benchmarks", "evaluate-momentum", pack, predictions])
    score = json.loads((args.output / "score.log").read_text())
    if score["all_tracks"]["p_mae_mev"] != 0 or score["all_tracks"]["pt_mae_mev"] != 0:
        raise ValueError("Perfect prediction scores are not zero")
    config = json.loads(args.config.read_text())
    config.update(steps_per_stage=4, validation_interval=2, checkpoint_validation_events=2, event_cache_size=1)
    if "warmup_steps" in config:
        config["warmup_steps"] = 1
    config["eggnet"].update(dbscan_eps=[.1, .5], dbscan_min_samples=[2])
    config["exatrkx"]["gnn_cuts"] = [.3, .7]
    recipe = args.output / "config.json"
    recipe.write_text(json.dumps(config))
    reports = {}
    if args.check_recovery:
        if args.device != "cpu":
            raise ValueError("Exact recovery comparison requires the deterministic CPU check")
        from clas12_benchmarks.prepare_cohort import prepare
        with prepare(root, args.output / "extended", reader="hdf5", train=counts[0], val=counts[1],
                     test=counts[2], seed=2026, extend_from=cohort.parent):
            pass
    for method in ("eggnet", "exatrkx"):
        output = args.output / method
        run(method, ["-m", "clas12_benchmarks.tracking", "--method", method, "--cohort", cohort,
                     "--config", recipe, "--metrics", args.metrics, "--output", output,
                     "--device", args.device, "--threads", 2])
        audit = args.output / f"{method}-audit.json"
        run(method + "-audit", [ROOT / "scripts/audit_tracking_predictions.py", "--run", output,
                               "--cohort", cohort, "--output", audit])
        frozen = args.output / (method + "-frozen")
        run(method + "-frozen", ["-m", "clas12_benchmarks.evaluate_frozen", "--parent", output,
                                 "--original-cohort", cohort, "--cohort", cohort, "--output", frozen,
                                 "--device", args.device, "--threads", 2])
        run(method + "-frozen-audit", [ROOT / "scripts/audit_tracking_predictions.py", "--run", frozen,
                                      "--cohort", cohort, "--output", args.output / f"{method}-frozen-audit.json"])
        before = json.loads((output / "summary.json").read_text())
        after = json.loads((frozen / "summary.json").read_text())
        for group in ("metrics", "coatjava"):
            for key, value in before[group].items():
                if value is None:
                    if after[group][key] is not None:
                        raise ValueError("Frozen undefined metric changed")
                else:
                    np.testing.assert_allclose(after[group][key], value, rtol=0, atol=1e-12)
        reports[method] = {**json.loads(audit.read_text()),
                           "resources": json.loads((output / "resources.json").read_text())}
        if args.check_recovery:
            resumed = args.output / (method + "-resumed")
            with (args.output / (method + "-recovery.log")).open("w") as log, redirect_stdout(log):
                check_resume(method, cohort, recipe, args.metrics, output, resumed)
            run(method + "-resumed-audit", [ROOT / "scripts/audit_tracking_predictions.py", "--run", resumed,
                                           "--cohort", cohort, "--output", args.output / f"{method}-resumed-audit.json"])
            extended = args.output / "extended/cohort.json"
            fresh = args.output / (method + "-fresh-holdout")
            run(method + "-fresh-holdout", ["-m", "clas12_benchmarks.evaluate_frozen", "--parent", output,
                "--original-cohort", cohort, "--cohort", extended, "--output", fresh, "--device", "cpu", "--threads", 2])
            run(method + "-fresh-audit", [ROOT / "scripts/audit_tracking_predictions.py", "--run", fresh,
                "--cohort", extended, "--output", args.output / f"{method}-fresh-audit.json"])
            bad = json.loads(cohort.read_text())
            bad["roles"]["train"].reverse()
            bad_path = args.output / "changed-training.json"
            bad_path.write_text(json.dumps(bad))
            from clas12_benchmarks.evaluate_frozen import evaluate_frozen
            try:
                evaluate_frozen(output, cohort, bad_path, args.output / (method + "-rejected-training"), device="cpu")
            except ValueError as exc:
                if "Training identities/order changed" not in str(exc):
                    raise
            else:
                raise ValueError("Frozen evaluation accepted changed training")
    with Cohort(cohort) as source:
        identity = {role: [source.event(role, i).identity for i in range(len(members))]
                    for role, members in source.roles.items()}
    report = {"passed": True, "device": args.device, "cohort_sha256": file_sha256(cohort),
              "identities": identity, "momentum": tracks.meta["selection_counts"], "audits": reports,
              "full_runner_recovery_checked": args.check_recovery}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "device": args.device, "audits": reports}))


if __name__ == "__main__":
    main()
