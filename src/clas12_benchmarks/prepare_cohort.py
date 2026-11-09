"""Freeze a random real-event subset without changing its parent dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mmap_ninja import RaggedMmap

from .data import Cohort, file_sha256
from .sources import open_ragged


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def draw_excluding(rng, population, count, excluded):
    excluded = set(excluded)
    if count > population - len(excluded):
        raise ValueError("Not enough unused source events")
    selected = []
    while len(selected) < count:
        batch = rng.choice(population, size=min(population, count - len(selected) + 100), replace=False)
        for value in batch:
            value = int(value)
            if value not in excluded:
                selected.append(value)
                excluded.add(value)
                if len(selected) == count:
                    break
    return selected


def prepare(root, output, train=10000, val=1000, test=2000, seed=1337, extend_from=None, reader="ragged_mmap",
            noise_policy="primary_unassigned_is_noise_v1"):
    if reader == "hdf5":
        from .hdf5_cohort import prepare as prepare_hdf5
        return prepare_hdf5(root, output, train, val, test, seed, extend_from, noise_policy)
    if reader != "ragged_mmap":
        raise ValueError(f"Unsupported reader: {reader}")
    if noise_policy != "primary_unassigned_is_noise_v1":
        raise ValueError("Selecting a noise policy requires the HDF5 reader")
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    metadata = {"schema": "clas12_benchmark_subset_v1", "source_root": str(root),
                "source_metadata_sha256": file_sha256(root / "metadata.json"),
                "seed": seed, "source_splits": {}, "selection": "uniform rows without replacement; no truth or size cuts"}
    previous = None
    preserved = {}
    if extend_from:
        previous = Cohort(Path(extend_from) / "cohort.json")
        previous_meta = json.loads((Path(extend_from) / "metadata.json").read_text())
        if previous_meta["source_metadata_sha256"] != metadata["source_metadata_sha256"]:
            raise ValueError("Parent source metadata changed")
        for role, members in previous.roles.items():
            preserved[role] = [previous_meta["source_splits"][previous.datasets[m["dataset"]][0]["source_split"]]["indices"][m["index"]]
                               for m in members]
        if train != len(preserved["train"]) or val < len(preserved["val"]):
            raise ValueError("Extension must preserve train and retain previous validation")
        metadata.update(parent_cohort_sha256=previous.sha256,
            selection="preserve training and old validation; expand validation uniformly; fresh test excludes all previous test events",
            previous_test_indices_excluded=preserved["test"])
    roles, datasets, identities = {}, {}, set()
    train_xyz = []
    for split, counts in (("pretrain", (train, val)), ("test", (test,))):
        features = open_ragged(root, "features", split)
        if previous:
            if split == "pretrain":
                new_val = draw_excluding(rng, len(features), val - len(preserved["val"]), preserved["train"] + preserved["val"])
                indices = preserved["train"] + preserved["val"] + new_val
            else:
                indices = draw_excluding(rng, len(features), test, preserved["test"])
        else:
            indices = rng.choice(len(features), size=sum(counts), replace=False)
        # Assign train/val before sorting; neither role is an index-range slice.
        assigned = {int(i): ("train" if j < train else "val") if split == "pretrain" else "test"
                    for j, i in enumerate(indices)}
        ordered = sorted(assigned)
        requested = set(ordered)
        rows = {}
        digest = hashlib.sha256()
        n_rows = 0
        with (root / f"samples_{split}.jsonl").open("rb") as stream:
            for i, line in enumerate(stream):
                digest.update(line)
                if i in requested:
                    row = json.loads(line)
                    identity = (row["source_file"], row["event"])
                    if row.get("sample_mode") != "event" or identity in identities:
                        raise ValueError(f"Non-event or duplicate identity: {identity}")
                    identities.add(identity)
                    rows[i] = row
                n_rows += 1
                if n_rows % 5000000 == 0:
                    print(f"{split}: scanned {n_rows:,}/{len(features):,} sample identities", flush=True)
        if n_rows != len(features) or len(rows) != len(ordered):
            raise ValueError("Sample identity count does not match native arrays")
        print(f"{split}: froze {len(ordered)} of {n_rows} events", flush=True)
        metadata["source_splits"][split] = {"n_events": n_rows, "indices": ordered,
            "samples_sha256": digest.hexdigest()}
        if previous and digest.hexdigest() != previous_meta["source_splits"][split]["samples_sha256"]:
            raise ValueError("Parent sample identities changed")
        arrays = {name: open_ragged(root, name, split) for name in
                  ("features", "seg_target", "coatjava_seg_pred")}
        for name, array in arrays.items():
            if len(array) != n_rows:
                raise ValueError(f"Array length mismatch: {name}")
            RaggedMmap.from_generator(output / f"{name}_{split}",
                (np.array(array[i], copy=True) for i in ordered), batch_size=1024, verbose=False)
        samples = output / f"samples_{split}.jsonl"
        with samples.open("w") as stream:
            for subset_index, source_index in enumerate(ordered):
                row = dict(rows[source_index], original_dataset_index=source_index)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                role = assigned[source_index]
                roles.setdefault(role, []).append({"dataset": split, "index": subset_index})
                if role == "train":
                    train_xyz.append(np.array(features[source_index], copy=True))
        datasets[split] = {"root": ".", "source_split": split, "input_layout": "xyz",
            "samples_sha256": file_sha256(samples), "extra_targets": ["coatjava_seg_pred"]}
    xyz = np.concatenate(train_xyz)
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    if not np.isfinite(xyz).all() or np.any(radius <= 0):
        raise ValueError("Invalid train coordinates")
    scales = {"eta_scale": max(float(np.max(np.abs(np.arcsinh(xyz[:, 2] / radius)))), 1e-6),
              "radius_scale": float(np.max(radius))}
    if previous:
        old_scales = previous_meta["normalization"]
        if any(scales[key] != old_scales[key] for key in scales):
            raise ValueError("Training normalization changed during extension")
    metadata["normalization"] = dict(scales, fitted_on="train only; max absolute eta and transverse radius")
    write_json(output / "metadata.json", metadata)
    for spec in datasets.values():
        spec.update(scales, metadata_sha256=file_sha256(output / "metadata.json"))
    write_json(output / "cohort.json", {"schema": "clas12_baseline_cohort_v1", "datasets": datasets, "roles": roles})
    cohort = Cohort(output / "cohort.json")
    summarize(cohort, output)
    return cohort


def summarize(cohort, output):
    summary = {}
    roles = cohort.roles
    for role, members in roles.items():
        events = [cohort.event(role, i) for i in range(len(members))]
        summary[role] = {"n_events": len(events), "n_points": sum(len(e.raw_features) for e in events),
            "n_sources": len({e.identity["source_file"] for e in events}),
            "point_count_range": [min(len(e.raw_features) for e in events), max(len(e.raw_features) for e in events)],
            "all_background_events": sum(bool(np.all(e.labels["seg_target"] < 0)) for e in events)}
    spec = next(iter(cohort.datasets.values()))[0]
    scales = {key: spec[key] for key in ("eta_scale", "radius_scale")}
    write_json(Path(output) / "summary.json", {"cohort_sha256": cohort.sha256, "roles": summary, "normalization": scales})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("output")
    p.add_argument("--train", type=int, default=10000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--reader", choices=("ragged_mmap", "hdf5"), default="ragged_mmap")
    p.add_argument("--noise-policy", choices=("primary_unassigned_is_noise_v1", "primary_valid_only_v1"),
                   default="primary_unassigned_is_noise_v1")
    p.add_argument("--extend-from", help="Previous prepared cohort directory; preserve train and use fresh test events")
    args = p.parse_args()
    with prepare(**vars(args)):
        pass
