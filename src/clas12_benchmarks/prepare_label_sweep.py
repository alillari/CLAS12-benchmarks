"""Nested label-budget cohorts with fixed holdouts and per-budget normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from mmap_ninja import RaggedMmap

from .data import Cohort, file_sha256
from .prepare_cohort import draw_excluding, write_json


def prepare(root, previous, output, budgets=(100, 1000, 10000, 100000), seed=1337):
    root, previous, output = map(Path, (root, previous, output))
    parent = Cohort(previous / "cohort.json")
    pm = json.loads((previous / "metadata.json").read_text())
    budgets = sorted(budgets)
    anchor = len(parent.roles["train"])
    if anchor not in budgets or len(set(budgets)) != len(budgets) or min(budgets) < 1:
        raise ValueError("Budgets must be unique positive sizes including the parent train size")
    if file_sha256(root / "metadata.json") != pm["source_metadata_sha256"]:
        raise ValueError("Source metadata changed")
    def original_indices(role):
        return [pm["source_splits"][parent.datasets[m["dataset"]][0]["source_split"]]["indices"][m["index"]]
                for m in parent.roles[role]]
    old_train, val = original_indices("train"), original_indices("val")
    arrays = {name: RaggedMmap(root / f"{name}_pretrain", mode="r")
              for name in ("features", "seg_target", "coatjava_seg_pred")}
    rng = np.random.default_rng(seed)
    extra = draw_excluding(rng, len(arrays["features"]), max(budgets) - anchor, old_train + val)
    order = list(map(int, np.random.default_rng(seed + 2).permutation(old_train))) + extra
    selected = sorted(order + val)
    requested = set(selected)
    rows, digest, count = {}, hashlib.sha256(), 0
    with (root / "samples_pretrain.jsonl").open("rb") as stream:
        for i, line in enumerate(stream):
            digest.update(line)
            if i in requested:
                rows[i] = json.loads(line)
            count += 1
            if count % 5000000 == 0:
                print(f"Scanned {count:,} sample identities", flush=True)
    if count != len(arrays["features"]) or len(rows) != len(selected):
        raise ValueError("Native sample/array counts differ")
    if digest.hexdigest() != pm["source_splits"]["pretrain"]["samples_sha256"]:
        raise ValueError("Parent source identities changed")
    output.mkdir(parents=True, exist_ok=False)
    for name, array in arrays.items():
        if len(array) != count:
            raise ValueError("Native target count mismatch")
        RaggedMmap.from_generator(output / f"{name}_pretrain", (np.array(array[i], copy=True) for i in selected), batch_size=1024)
    with (output / "samples_pretrain.jsonl").open("w") as stream:
        for i in selected:
            stream.write(json.dumps(dict(rows[i], original_dataset_index=i), sort_keys=True) + "\n")
    for name in ("features", "seg_target", "coatjava_seg_pred"):
        shutil.copytree(previous / f"{name}_test", output / f"{name}_test")
    shutil.copyfile(previous / "samples_test.jsonl", output / "samples_test.jsonl")
    metadata = {"schema": "clas12_label_sweep_data_v1", "seed": seed,
        "source_metadata_sha256": pm["source_metadata_sha256"], "parent_cohort_sha256": parent.sha256,
        "source_splits": {"pretrain": {"indices": selected, "samples_sha256": digest.hexdigest(), "n_events": count},
                          "test": pm["source_splits"]["test"]},
        "normalization_policy": "fit maximum absolute eta and radius separately on each selected training budget",
        "label_budget_unit": "fully labeled training events; excludes fixed validation labels",
        "nesting_order_source_indices": order}
    write_json(output / "metadata.json", metadata)
    location = {source: i for i, source in enumerate(selected)}
    summary = {"schema": "clas12_label_sweep_v1", "parent_cohort_sha256": parent.sha256,
               "seed": seed, "budgets": {}, "fixed_validation_events": len(val),
               "fixed_test_events": len(parent.roles["test"])}
    for budget in budgets:
        sources = sorted(order[:budget])
        xyz = np.concatenate([np.array(arrays["features"][i], copy=True) for i in sources])
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        if not np.isfinite(xyz).all() or np.any(radius <= 0):
            raise ValueError("Invalid training geometry")
        scales = {"eta_scale": max(float(np.abs(np.arcsinh(xyz[:, 2] / radius)).max()), 1e-6),
                  "radius_scale": float(radius.max())}
        datasets = {split: {"root": ".", "source_split": split, "input_layout": "xyz", **scales,
            "metadata_sha256": file_sha256(output / "metadata.json"),
            "samples_sha256": file_sha256(output / f"samples_{split}.jsonl"),
            "extra_targets": ["coatjava_seg_pred"]} for split in ("pretrain", "test")}
        roles = {"train": [{"dataset": "pretrain", "index": location[i]} for i in sources],
                 "val": [{"dataset": "pretrain", "index": location[i]} for i in val],
                 "test": [{"dataset": "test", "index": m["index"]} for m in parent.roles["test"]]}
        manifest = output / f"cohort_{budget}.json"
        write_json(manifest, {"schema": "clas12_baseline_cohort_v1", "datasets": datasets, "roles": roles})
        cohort = Cohort(manifest)
        events = [cohort.event("train", i) for i in range(budget)]
        summary["budgets"][str(budget)] = {"manifest": manifest.name, "cohort_sha256": cohort.sha256,
            "n_training_events": budget, "n_training_points": len(xyz),
            "n_training_tracks": sum(len(np.unique(e.labels["seg_target"][e.labels["seg_target"] >= 0])) for e in events),
            "n_training_sources": len({e.identity["source_file"] for e in events}),
            "all_background_training_events": sum(bool(np.all(e.labels["seg_target"] < 0)) for e in events),
            "normalization": scales}
        print(json.dumps(summary["budgets"][str(budget)]), flush=True)
    write_json(output / "label_sweep_summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("previous")
    p.add_argument("output")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    prepare(args.root, args.previous, args.output, seed=args.seed)
