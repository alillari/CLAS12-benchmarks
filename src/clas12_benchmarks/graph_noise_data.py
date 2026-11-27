"""Freeze whole-event noise samples from the published cluster splits."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .data import file_sha256
from .hdf5 import ClusterSource, MIN4_NOISE_POLICY, POINT_FIELDS, cluster_projection
from .hdf5_cohort import SPLITS, fm_cohort_spec, publication, write_json

GEOMETRY_FIELDS = ("clusters.detector", "clusters.layer")


def freeze(root, output, counts, seed, fm_cohorts=None, plan_id=None):
    """Write one compact array archive per split and a source/identity receipt."""
    if (fm_cohorts is None) != (plan_id is None):
        raise ValueError("Provide both the FM event roster and plan ID")
    with ExitStack() as stack, publication(output) as stage:
        sources = {role: stack.enter_context(ClusterSource(root, split))
                   for role, split in SPLITS.items()}
        rng = np.random.default_rng(seed)
        if fm_cohorts is None:
            members = {role: rng.choice(len(source), count, replace=False)
                       for (role, source), count in zip(sources.items(), counts, strict=True)}
        else:
            import json
            spec = fm_cohort_spec(root, json.loads(Path(fm_cohorts).read_text()), plan_id,
                                  noise_policy=MIN4_NOISE_POLICY)
            members = {role: [row["index"] for row in rows] for role, rows in spec["roles"].items()}
        seen = set()
        receipts = {}
        for role, source in sources.items():
            indices = members[role]
            identities = source.identities(indices)
            address_fields = [name for name in GEOMETRY_FIELDS
                              if name.split(".")[1] in source.schema["fields"]["clusters"]]
            arrays = {}
            for row, index in sorted(enumerate(indices), key=lambda pair: pair[1]):
                identity = identities[index]
                key = source.overlap_key(identity)
                if key in seen:
                    raise ValueError(f"Overlapping physical event: {role}:{index}")
                seen.add(key)
                values = source.read(index, (*POINT_FIELDS, *address_fields))
                xyz = values["clusters.xyz"]
                if not np.isfinite(xyz).all() or np.any(np.hypot(xyz[:, 0], xyz[:, 1]) <= 0):
                    raise ValueError(f"Invalid cluster centers at {role}:{index}")
                labels, _ = cluster_projection(values, noise_policy=MIN4_NOISE_POLICY)
                reco = labels["coatjava_seg_pred"]
                arrays[row] = (xyz, labels["noise_target"].astype(np.int8), reco < 0,
                               *(values.get(name, np.zeros(len(xyz), dtype=np.int16)).astype(np.int16)
                                 for name in GEOMETRY_FIELDS))
            events = [arrays[row] for row in range(len(indices))]
            sizes = np.asarray([len(event[0]) for event in events])
            path = stage / f"{role}.npz"
            columns = dict(zip(("xyz", "labels", "coatjava_noise", "detector", "layer"),
                               map(np.concatenate, zip(*events, strict=True)), strict=True))
            np.savez(path, **columns, offsets=np.r_[0, sizes.cumsum()],
                     source_event_index=indices,
                     source_id=[identities[i]["source_id"] for i in indices],
                     original_ordinal=[identities[i]["original_ordinal"] for i in indices])
            receipts[role] = dict(contract=source.contract, events=len(events), clusters=int(sizes.sum()),
                                  detector_layer_available=len(address_fields) == 2,
                                  minimum_occupancy=int(sizes.min()), maximum_occupancy=int(sizes.max()),
                                  sha256=file_sha256(path))
            print(f"Prepared {role}: {len(events)} complete events, {sizes.sum()} clusters", flush=True)
        write_json(stage / "cohort.json",
                   dict(schema="clas12_cluster_graph_noise_cache_v1", source_root=str(root),
                        noise_policy=MIN4_NOISE_POLICY, seed=seed, splits=receipts,
                        selection=("imported exact FM evaluation roster" if fm_cohorts else
                                   "uniform without replacement within each published split"),
                        fm_cohorts_sha256=file_sha256(fm_cohorts) if fm_cohorts else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--validation", type=int, default=2400)
    parser.add_argument("--test", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--fm-cohorts", type=Path)
    parser.add_argument("--plan-id")
    args = parser.parse_args()
    counts = (args.train, args.validation, args.test)
    if min(counts) < 1 or args.seed < 0:
        parser.error("Split sizes must be positive and seed nonnegative")
    freeze(args.data_root, args.output, counts, args.seed, args.fm_cohorts, args.plan_id)


if __name__ == "__main__":
    main()
