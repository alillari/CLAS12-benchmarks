"""Import exact FM HDF5 event membership, preserving its role order and cuts."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .data import Cohort, file_sha256
from .finding_evaluation import FindingEvaluation, contract, finding_sequence
from .hdf5 import ClusterSource
from .hdf5_cohort import SPLITS, cohort_spec, normalization, publication, write_json


def prepare(data_root, fm_cohorts, output, plan_id):
    """The FM cohorts.json supplies source_id, original_ordinal, source_event_index.

    Require an explicit expected publication plan ID because those integer event
    keys alone cannot establish which HDF5 publication produced an FM report.
    """
    fm_cohorts = Path(fm_cohorts)
    saved = json.loads(fm_cohorts.read_text())
    if set(saved) != {"train", "validation", "test"}:
        raise ValueError("Expected nonempty FM train/validation/test event cohorts")
    members = {}
    with ExitStack() as stack:
        sources = {split: stack.enter_context(ClusterSource(data_root, split)) for split in SPLITS.values()}
        if any(source.manifest["plan_id"] != plan_id for source in sources.values()):
            raise ValueError("FM HDF5 plan_id mismatch")
        for role, split in SPLITS.items():
            rows = saved[split]
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Empty or invalid FM {split} cohort")
            indices = []
            for row in rows:
                index = row.get("source_event_index")
                if type(index) is not int or index < 0 or row.get("segment_label") is not None:
                    raise ValueError("FM finding cohorts must contain complete event samples")
                if any(type(row.get(key)) is not int for key in ("source_id", "original_ordinal")):
                    raise ValueError("Missing FM physical event identity")
                indices.append(index)
            identities = sources[split].identities(indices)
            for row, index in zip(rows, indices, strict=True):
                identity = identities[index]
                for key in ("source_id", "original_ordinal", "source_manifest_id"):
                    if (key != "source_manifest_id" or key in row) and row.get(key) != identity[key]:
                        raise ValueError(f"FM physical event identity mismatch: {split}:{index}")
            if len(set(indices)) != len(indices):
                raise ValueError(f"Duplicate FM {split} events")
            members[role] = indices
        spec = cohort_spec(sources, members, normalization(sources["train"], members["train"]))
        spec["finding_evaluation"] = contract(new_run=True)
        evaluation = FindingEvaluation(settings=spec["finding_evaluation"])
        with publication(output) as stage:
            write_json(stage / "cohort.json", spec)
            with Cohort(stage / "cohort.json") as cohort:
                populations = {}
                for role in SPLITS:
                    events = finding_sequence(cohort, role, evaluation)
                    populations[role] = {"n_events": len(events),
                        "n_clusters": sum(len(e.raw_features) for e in events),
                        "n_eligible_tracks": sum(len(np.unique(e.labels["seg_target"][e.labels["seg_target"] >= 0])) for e in events)}
                write_json(stage / "summary.json", {**evaluation.metadata(), "roles": populations,
                                                    "cohort_sha256": cohort.sha256})
            write_json(stage / "metadata.json", {"schema": "clas12_fm_finding_cohort_import_v1",
                "fm_cohorts": str(fm_cohorts.resolve()), "fm_cohorts_sha256": file_sha256(fm_cohorts),
                "plan_id": plan_id, "selection": "exact FM event membership and order; no resampling or additional cuts",
                "normalization": "fitted on imported training coordinates only"})
    return Cohort(Path(output) / "cohort.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fm-cohorts", required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with prepare(**vars(args)):
        pass


if __name__ == "__main__":
    main()
