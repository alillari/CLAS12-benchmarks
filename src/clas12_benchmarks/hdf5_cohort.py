"""Freeze native HDF5 event cohorts without copying the raw shards."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data import Cohort, source_key
from .hdf5 import ClusterSource, COHORT_SCHEMA, FORMAT, FEATURE_RECIPE, NOISE_POLICY, PID_POLICY

SPLITS = {"train": "train", "val": "validation", "test": "test"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@contextmanager
def publication(output):
    """Reserve a new directory; publish cohort JSON last, clean up on failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
            stage = Path(temporary)
            yield stage
            files = sorted(stage.iterdir(), key=lambda p: p.name == "cohort.json")
            for path in files:
                os.link(path, output / path.name)
    except BaseException:
        shutil.rmtree(output)
        raise


def normalization(source, indices):
    eta_scale, radius_scale = 1e-6, 0.0
    for index in indices:
        xyz = source.read(index, ["clusters.xyz"])["clusters.xyz"]
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        if (xyz.shape != (len(xyz), 3) or not np.isfinite(xyz).all() or np.any(radius <= 0)):
            raise ValueError(f"Invalid training coordinates at {source.split}:{index}")
        if len(radius):
            eta_scale = max(eta_scale, float(np.abs(np.arcsinh(xyz[:, 2] / radius)).max()))
            radius_scale = max(radius_scale, float(radius.max()))
    if radius_scale <= 0:
        raise ValueError("Training cohort has no usable coordinates")
    return {"eta_scale": eta_scale, "radius_scale": radius_scale}


def cohort_spec(sources, members, scales, noise_policy=NOISE_POLICY):
    datasets, roles = {}, {}
    for role, split in SPLITS.items():
        source = sources[split]
        indices = list(map(int, members[role]))
        datasets[split] = {"root": str(source.root), "source_split": split,
                           "storage_format": FORMAT, "token_kind": "clusters", "input_layout": "xyz",
                           "feature_recipe": FEATURE_RECIPE, "noise_policy": noise_policy,
                           "pid_policy": PID_POLICY, "extra_targets": ["coatjava_seg_pred", "pid_target", "noise_target"],
                           "source_contract": source.contract,
                           "selected_content_sha256": source.content_digest(indices), **scales}
        roles[role] = [{"dataset": split, "index": index} for index in indices]
    return {"schema": COHORT_SCHEMA, "datasets": datasets, "roles": roles}


def fm_cohort_spec(root, saved, plan_id, *, noise_policy=NOISE_POLICY):
    """Freeze FM event membership and order against the original HDF5 publication."""
    if not isinstance(saved, dict) or set(saved) != set(SPLITS.values()):
        raise ValueError("Expected nonempty FM train/validation/test event cohorts")
    members = {}
    with ExitStack() as stack:
        sources = {split: stack.enter_context(ClusterSource(root, split)) for split in SPLITS.values()}
        if any(source.manifest["plan_id"] != plan_id for source in sources.values()):
            raise ValueError("FM HDF5 plan_id mismatch")
        for role, split in SPLITS.items():
            rows = saved[split]
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Empty or invalid FM {split} cohort")
            indices = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid FM {split} event")
                index = row.get("source_event_index")
                if type(index) is not int or index < 0 or row.get("segment_label") is not None:
                    raise ValueError("FM cohorts must contain complete event samples")
                if any(type(row.get(key)) is not int for key in ("source_id", "original_ordinal")):
                    raise ValueError("Missing FM physical event identity")
                indices.append(index)
            if len(set(indices)) != len(indices):
                raise ValueError(f"Duplicate FM {split} events")
            identities = sources[split].identities(indices)
            for row, index in zip(rows, indices, strict=True):
                identity = identities[index]
                keys = ("source_id", "original_ordinal", "source_manifest_id")
                if any(key in row and row[key] != identity[key] for key in keys):
                    raise ValueError(f"FM physical event identity mismatch: {split}:{index}")
            members[role] = indices
        scales = normalization(sources["train"], members["train"])
        return cohort_spec(sources, members, scales, noise_policy)


def summary(cohort):
    roles = {}
    for role, members in cohort.roles.items():
        count, low, high, background, sources = 0, None, 0, 0, set()
        for i in range(len(members)):
            event = cohort.event(role, i)
            n = len(event.raw_features)
            count += n
            low = n if low is None else min(low, n)
            high = max(high, n)
            background += int(np.all(event.labels["seg_target"] < 0))
            sources.add(source_key(event.identity))
        roles[role] = {"n_events": len(members), "n_points": count, "n_sources": len(sources),
                       "point_count_range": [low, high], "all_background_events": background}
    return {"cohort_sha256": cohort.sha256, "roles": roles,
            "normalization": {key: next(iter(cohort.datasets.values()))[0][key]
                              for key in ("eta_scale", "radius_scale")}}


def _indices(cohort, role):
    return [member["index"] for member in cohort.roles[role]]


def validate_parent_sources(sources, parent):
    """A relocated publication must preserve the parent's actual selected bytes."""
    for name, (spec, rows, _) in parent.datasets.items():
        source = sources[spec["source_split"]]
        if source.contract != spec["source_contract"]:
            raise ValueError("Parent source contract changed")
        if source.content_digest(rows) != spec["selected_content_sha256"]:
            raise ValueError(f"Parent selected source content changed: {name}")


def prepare(root, output, train, val, test, seed, extend_from=None, noise_policy=NOISE_POLICY):
    from .prepare_cohort import draw_excluding
    counts = {"train": train, "val": val, "test": test}
    if any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("All cohort counts must be positive integers")
    with ExitStack() as stack:
        sources = {split: stack.enter_context(ClusterSource(root, split)) for split in SPLITS.values()}
        parent = stack.enter_context(Cohort(Path(extend_from) / "cohort.json")) if extend_from else None
        if parent:
            if not parent.is_hdf5:
                raise ValueError("Cannot extend a legacy cohort with an HDF5 source")
            validate_parent_sources(sources, parent)
            for name, (spec, _, _) in parent.datasets.items():
                if spec["noise_policy"] != noise_policy:
                    raise ValueError("Parent noise policy changed")
            if train != len(parent.roles["train"]) or val < len(parent.roles["val"]):
                raise ValueError("Extension must preserve train and retain validation")
        rng, members = np.random.default_rng(seed), {}
        for role, split in SPLITS.items():
            old = _indices(parent, role) if parent else []
            if parent and role == "train":
                members[role] = old
            elif parent and role == "val":
                members[role] = old + draw_excluding(rng, len(sources[split]), val - len(old), old)
            else:
                members[role] = sorted(draw_excluding(rng, len(sources[split]), counts[role], old))
        scales = normalization(sources["train"], members["train"])
        if parent and any(scales[k] != next(iter(parent.datasets.values()))[0][k] for k in scales):
            raise ValueError("Training normalization changed")
        spec = cohort_spec(sources, members, scales, noise_policy)
        with publication(output) as stage:
            write_json(stage / "cohort.json", spec)
            with Cohort(stage / "cohort.json") as cohort:
                write_json(stage / "summary.json", summary(cohort))
            write_json(stage / "metadata.json", {"schema": "clas12_hdf5_subset_v1", "seed": seed,
                       "source_root": str(Path(root).resolve()), "normalization": scales,
                       "parent_cohort_sha256": parent.sha256 if parent else None,
                       "selection": "uniform events per published split; no truth or size cuts"})
    return Cohort(Path(output) / "cohort.json")


def prepare_sweep(root, previous, output, budgets, seed):
    from .prepare_cohort import draw_excluding
    budgets = sorted(budgets)
    with ExitStack() as stack:
        parent = stack.enter_context(Cohort(Path(previous) / "cohort.json"))
        if not parent.is_hdf5:
            raise ValueError("HDF5 sweep requires an HDF5 parent")
        anchor = len(parent.roles["train"])
        if (not budgets or any(type(v) is not int or v < 1 for v in budgets)
                or len(set(budgets)) != len(budgets) or anchor not in budgets):
            raise ValueError("Budgets must be unique positive integers including the parent size")
        sources = {split: stack.enter_context(ClusterSource(root, split)) for split in SPLITS.values()}
        validate_parent_sources(sources, parent)
        old = _indices(parent, "train")
        extra = draw_excluding(np.random.default_rng(seed), len(sources["train"]), max(budgets) - anchor, old)
        order = list(map(int, np.random.default_rng(seed + 2).permutation(old))) + extra
        report = {"schema": "clas12_label_sweep_v1", "parent_cohort_sha256": parent.sha256,
                  "seed": seed, "budgets": {}, "fixed_validation_events": len(parent.roles["val"]),
                  "fixed_test_events": len(parent.roles["test"])}
        with publication(output) as stage:
            for budget in budgets:
                members = {"train": sorted(order[:budget]), "val": _indices(parent, "val"),
                           "test": _indices(parent, "test")}
                scales = normalization(sources["train"], members["train"])
                spec = cohort_spec(sources, members, scales,
                                   next(iter(parent.datasets.values()))[0]["noise_policy"])
                filename = f"cohort_{budget}.json"
                write_json(stage / filename, spec)
                with Cohort(stage / filename) as cohort:
                    counts = summary(cohort)["roles"]["train"]
                    tracks = 0
                    for i in range(budget):
                        labels = cohort.event("train", i).labels["seg_target"]
                        tracks += len(np.unique(labels[labels >= 0]))
                    report["budgets"][str(budget)] = {"manifest": filename, "cohort_sha256": cohort.sha256,
                        "n_training_events": budget, "n_training_points": counts["n_points"],
                        "n_training_tracks": tracks, "n_training_sources": counts["n_sources"],
                        "all_background_training_events": counts["all_background_events"], "normalization": scales}
            write_json(stage / "metadata.json", {"schema": "clas12_hdf5_label_sweep_v1",
                       "nesting_order_source_indices": order, "parent_cohort_sha256": parent.sha256})
            write_json(stage / "label_sweep_summary.json", report)
    return report
