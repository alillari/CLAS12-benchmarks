#!/usr/bin/env python3
"""Audit release occupancy, raw cluster fields and cross-release source identities."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from clas12_benchmarks.graph_noise import Split
from clas12_benchmarks.hdf5 import ClusterSource, MIN4_NOISE_POLICY, POINT_FIELDS, cluster_projection
from clas12_benchmarks.hdf5_cohort import write_json


def audit(data_root, cohort, historical):
    manifest = json.loads((data_root / "manifest.json").read_text())
    receipt = json.loads((cohort / "cohort.json").read_text())
    report = {"dataset_plan_id": manifest["plan_id"], "splits": {}}
    for role, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        frequencies = np.zeros(1, dtype=np.int64)
        top = None
        for part in manifest["splits"][split]:
            with h5py.File(data_root / part["path"], "r") as handle:
                if (handle.attrs["events"] != part["events"] or handle.attrs["complete"] != 1
                        or handle.attrs["plan_id"] != manifest["plan_id"]
                        or handle.attrs["format"] != manifest["format"]
                        or handle.attrs["part_key"] != part["path"]):
                    raise ValueError(f"Shard attributes changed: {part['path']}")
                offsets = handle["clusters/offsets"][:]
                points = handle["clusters/xyz"].shape[0]
                counts = np.diff(offsets)
                if (len(counts) != part["events"] or offsets[0] != 0 or offsets[-1] != points
                        or np.any(counts < 0)):
                    raise ValueError(f"Invalid cluster offsets: {part['path']}")
                hist = np.bincount(counts)
                if len(hist) > len(frequencies):
                    frequencies = np.pad(frequencies, (0, len(hist) - len(frequencies)))
                frequencies[:len(hist)] += hist
                if top is None or counts.max() > top["clusters"]:
                    top = dict(clusters=int(counts.max()), path=part["path"],
                               part_event=int(counts.argmax()), source_event_index=int(frequencies.sum() - len(counts) + counts.argmax()))
        qs = [.5, .9, .95, .99, .999, 1.]
        report["splits"][split] = dict(events=int(frequencies.sum()),
                                       clusters=int(frequencies @ np.arange(len(frequencies))),
                                       counts_by_occupancy=frequencies.tolist(),
                                       quantile_probabilities=qs,
                                       quantile_counts=np.searchsorted(frequencies.cumsum(),
                                                                      np.ceil(np.array(qs) * frequencies.sum())).tolist(),
                                       max_event=top)
        print(f"Audited {split} shard headers and offsets: {frequencies.sum()} events", flush=True)

    train = Split(cohort, receipt, "train")
    fields = ("clusters.detector", "clusters.layer", "clusters.endpoints", "clusters.seed_strip")
    geometry, n_baseline_changed, n_unknown = {}, 0, 0
    with ClusterSource(data_root, "train") as source:
        for i in sorted(train.rows["source_event_index"][:1024]):
            values = source.read(i, (*POINT_FIELDS, *fields))
            labels, _ = cluster_projection(values, noise_policy=MIN4_NOISE_POLICY)
            native_valid = values["clusters.primary_noise_valid"]
            n_unknown += int((~native_valid).sum())
            n_baseline_changed += int(np.count_nonzero(labels["noise_target"][native_valid] !=
                                                       values["clusters.primary_noise"][native_valid]))
            for code, name in ((1, "BMT"), (2, "BST")):
                active = values["clusters.detector"] == code
                if not active.any():
                    continue
                entry = geometry.setdefault(name, dict(clusters=0, endpoints_finite=0,
                                                       nonzero_span=0, seed_strip_min=None,
                                                       seed_strip_max=None, layers=set()))
                endpoints = values["clusters.endpoints"][active]
                finite = np.isfinite(endpoints).all(axis=1)
                entry["clusters"] += len(endpoints)
                entry["endpoints_finite"] += int(finite.sum())
                entry["nonzero_span"] += int(np.any(endpoints[finite, :3] != endpoints[finite, 3:], axis=1).sum())
                seeds = values["clusters.seed_strip"][active]
                for key, choice in (("seed_strip_min", min), ("seed_strip_max", max)):
                    local = int(choice(seeds))
                    entry[key] = local if entry[key] is None else choice(entry[key], local)
                entry["layers"].update(values["clusters.layer"][active].tolist())
    for entry in geometry.values():
        entry["layers"] = sorted(entry["layers"])
    report["training_geometry_probe"] = dict(events=1024, geometry=geometry,
                                             native_noise_invalid=n_unknown,
                                             valid_native_noise_changed_by_truth_policy=n_baseline_changed)
    if historical is not None:
        previous = json.loads((historical / "manifest.json").read_text())

        def identities(table, split, field, kind):
            source = (row[field] for row in table["sources"] if row["split"] == split and row.get(field))
            if kind == "basename":
                return {Path(row["path"]).name for row in source}
            return {row["sha256"] for row in source}

        report["historical_train_vs_v2"] = {
            split: {f"{field}_{kind}": len(identities(previous, "train", field, kind) &
                                           identities(manifest, split, field, kind))
                    for field, kind in (("signal", "basename"), ("signal", "sha256"),
                                        ("signal_raw", "sha256"), ("background", "sha256"))}
            for split in manifest["splits"]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--historical", type=Path)
    options = parser.parse_args()
    write_json(options.output, audit(options.data_root, options.cohort, options.historical))
