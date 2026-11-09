"""Observed direct-primary momentum packs with explicit native reco association."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import Cohort, event_key, file_sha256
from .hdf5 import POINT_FIELDS, update_digest

SCHEMA = "clas12_momentum_tracks_v2"
IDENTITY_FIELDS = ("source_manifest_id", "source_id", "original_ordinal", "primary_tid")


def _reference(rows, tid, domain, pass_id, min_purity):
    prefix = "reco." + domain + "."
    evidence = "reco.contributors."
    n = len(rows[prefix + "native"])
    candidates = np.unique(rows[evidence + "track_row"][
        (rows[evidence + "tid"] == tid) & (rows[evidence + "pass_id"] == pass_id)
        & (rows[evidence + "count"] > 0)])
    if np.any(candidates < 0) or np.any(candidates >= n):
        raise ValueError("Contributor candidate row outside same-pass track collection")
    qualified = []
    for row in candidates:
        if (rows[prefix + "purity_tid"][row] != tid or not rows[prefix + "purity_valid"][row]
                or not rows[prefix + "unique_owner"][row]
                or not np.isfinite(rows[prefix + "purity"][row])
                or rows[prefix + "purity"][row] < min_purity):
            continue
        p, theta, phi = (float(rows[prefix + key][row]) for key in
                         ("benchmark_p", "benchmark_theta", "entrance_phi"))
        entrance = int(rows[prefix + "entrance_row"][row])
        if not (np.isfinite([p, theta, phi]).all() and p > 0 and 0 <= theta <= np.pi and entrance >= 0):
            continue
        trajectory = rows["reco." + domain.replace("_tracks", "_trajectory") + ".native"]
        if entrance >= len(trajectory):
            raise ValueError("Entrance row outside same-pass trajectory collection")
        if int(trajectory["id"][entrance]) != int(rows[prefix + "native"]["ID"][row]):
            raise ValueError("Entrance trajectory and native track IDs disagree")
        direction = trajectory[entrance]
        if (not np.isfinite([direction["cx"], direction["cy"]]).all()
                or np.hypot(direction["cx"], direction["cy"]) <= 0):
            continue
        expected_phi = np.arctan2(direction["cy"], direction["cx"])
        difference = np.arctan2(np.sin(phi - expected_phi), np.cos(phi - expected_phi))
        if abs(difference) > 2e-6:
            raise ValueError("Stored entrance phi disagrees with same-pass trajectory")
        pt = p * np.sin(theta)
        qualified.append([pt * np.cos(phi), pt * np.sin(phi), p * np.cos(theta)])
    if len(qualified) == 1:
        return qualified[0], "valid"
    return [np.nan] * 3, "ambiguous" if len(qualified) > 1 else "missing_or_unqualified"


def build_manifest(cohort_path, destination, *, min_clusters=12, max_clusters=100,
                   cvt_pass="cvt", min_cvt_purity=1.0, train_track_budgets=()):
    if (type(min_clusters) is not int or type(max_clusters) is not int
            or not 1 <= min_clusters <= max_clusters or cvt_pass not in {"cvt", "cvtrec"}
            or not np.isfinite(min_cvt_purity) or not 0 <= min_cvt_purity <= 1):
        raise ValueError("Invalid momentum selection/reference recipe")
    domain = cvt_pass + "_tracks"
    reco_fields = ["reco." + domain + "." + key for key in (
        "native", "benchmark_p", "benchmark_theta", "entrance_phi", "entrance_row",
        "purity_tid", "purity_valid", "unique_owner", "purity")]
    fields = [*POINT_FIELDS, "particles.observed_momentum", "particles.observed_valid",
              *reco_fields, "reco." + cvt_pass + "_trajectory.native",
              *("reco.contributors." + k for k in ("pass_id", "track_row", "tid", "count"))]
    output = {name: [] for name in (*IDENTITY_FIELDS, "segment_label", "source_event_index",
                                    "truth_gev", "cvt_gev", "xyz_cm", "point_index", "lengths")}
    sizes, reports, digest = {}, {}, hashlib.sha256()
    with Cohort(cohort_path) as cohort:
        if not cohort.is_hdf5:
            raise ValueError("HDF5 momentum preparation requires a v2 HDF5 cohort")
        for input_role, role in (("train", "train"), ("val", "validation"), ("test", "evaluation")):
            counts = {"events": len(cohort.roles[input_role]), "segments": 0, "length_excluded": 0,
                      "invalid_truth": 0, "accepted": 0, "cvt_valid": 0,
                      "cvt_ambiguous": 0, "cvt_missing_or_unqualified": 0}
            for member in cohort.roles[input_role]:
                source = cohort.sources[member["dataset"]]
                index = member["index"]
                identity = source.identities([index])[index]
                rows = source.read(index, fields)
                digest.update(json.dumps([role, event_key(identity)]).encode())
                for field, array in rows.items():
                    update_digest(digest, field, array)
                tids = rows["particles.tid"]
                if np.any(tids <= 0) or len(np.unique(tids)) != len(tids):
                    raise ValueError("Particle tids must be unique and positive")
                owners = rows["clusters.primary_tid"]
                if np.any((owners != -1) & (owners <= 0)) or not np.isin(owners[owners > 0], tids).all():
                    raise ValueError("Positive cluster owner absent from event particles")
                seg, xyz = rows["clusters.primary_segment"], rows["clusters.xyz"]
                if owners.shape != seg.shape or xyz.shape != (len(seg), 3) or not np.isfinite(xyz).all():
                    raise ValueError("Invalid cluster point alignment")
                if np.any((seg < 0) != (owners < 0)):
                    raise ValueError("Segment and primary-owner availability disagree")
                used = set()
                for label in np.unique(seg[seg >= 0]):
                    counts["segments"] += 1
                    indices = np.flatnonzero(seg == label)
                    owner = np.unique(owners[indices])
                    if len(owner) != 1 or owner[0] <= 0 or int(owner[0]) in used:
                        raise ValueError("Conflicting segment/particle association")
                    tid = int(owner[0]); used.add(tid)
                    if not min_clusters <= len(indices) <= max_clusters:
                        counts["length_excluded"] += 1
                        continue
                    row = int(np.flatnonzero(tids == tid)[0])
                    truth = rows["particles.observed_momentum"][row]
                    valid = rows["particles.observed_valid"][row]
                    if not valid.all() or not np.isfinite(truth).all():
                        counts["invalid_truth"] += 1
                        continue
                    cvt, reason = _reference(rows, tid, domain, 0 if cvt_pass == "cvt" else 1, min_cvt_purity)
                    counts["accepted"] += 1
                    counts["cvt_" + reason] += 1
                    for key in IDENTITY_FIELDS[:-1]:
                        output[key].append(identity[key])
                    output["primary_tid"].append(tid)
                    output["segment_label"].append(int(label))
                    output["source_event_index"].append(index)
                    output["truth_gev"].append(truth.astype(np.float64))
                    output["cvt_gev"].append(cvt)
                    output["xyz_cm"].append(xyz[indices])
                    output["point_index"].append(indices)
                    output["lengths"].append(len(indices))
            sizes[role], reports[role] = counts["accepted"], counts
        metadata = {"schema": SCHEMA, "position_unit": "cm", "momentum_unit": "GeV",
                    "target": "observed_direct_primary_momentum", "source_cohort_sha256": cohort.sha256,
                    "source_contracts": {name: spec["source_contract"] for name, (spec, _, _) in cohort.datasets.items()},
                    "consumed_content_sha256": digest.hexdigest(), "min_clusters": min_clusters,
                    "max_clusters": max_clusters, "cvt_pass": cvt_pass, "min_cvt_purity": min_cvt_purity,
                    "cvt_policy": "single_qualified_unique_owner_same_pass_v1",
                    "role_sizes": sizes, "selection_counts": reports}
    arrays = {}
    for name, values in output.items():
        if name == "lengths":
            continue
        if name in {"xyz_cm", "point_index"}:
            arrays[name] = np.concatenate(values) if values else np.empty((0, 3) if name == "xyz_cm" else (0,),
                                                                         dtype=np.float32 if name == "xyz_cm" else np.int64)
        elif name in {"truth_gev", "cvt_gev"}:
            arrays[name] = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        else:
            arrays[name] = np.asarray(values, dtype=str if name == "source_manifest_id" else np.int64)
    arrays["offsets"] = np.r_[0, np.cumsum(output["lengths"], dtype=np.int64)]
    budgets = sorted(train_track_budgets)
    if (len(set(budgets)) != len(budgets) or any(type(b) is not int or b < 1 or b > sizes["train"] for b in budgets)):
        raise ValueError("Training track budgets must be unique positive sizes within the frozen training population")
    metadata["train_track_budgets"] = budgets
    metadata["array_sha256"] = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent) as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        stream.flush()
        # Validate before publishing; an invalid pack never gets its final name.
        PackedTracks(stream.name)
        os.link(stream.name, destination)
    return metadata


class PackedTracks:
    identity_fields = IDENTITY_FIELDS

    def __init__(self, path):
        self.sha256 = file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            self.meta = json.loads(str(archive["metadata"]))
            if (self.meta.get("schema") != SCHEMA or self.meta.get("position_unit") != "cm"
                    or self.meta.get("momentum_unit") != "GeV"):
                raise ValueError("Unsupported HDF5 momentum pack/units")
            required = {*IDENTITY_FIELDS, "segment_label", "source_event_index", "truth_gev", "cvt_gev",
                        "xyz_cm", "point_index", "offsets"}
            if (set(self.meta.get("array_sha256", {})) != required
                    or set(archive.files) != required | {"metadata"}
                    or self.meta.get("target") != "observed_direct_primary_momentum"
                    or type(self.meta.get("min_clusters")) is not int
                    or type(self.meta.get("max_clusters")) is not int
                    or not 1 <= self.meta["min_clusters"] <= self.meta["max_clusters"]):
                raise ValueError("Invalid HDF5 momentum pack field/target contract")
            for name, expected in self.meta["array_sha256"].items():
                value = archive[name]
                if hashlib.sha256(value.tobytes()).hexdigest() != expected:
                    raise ValueError(f"Packed array content mismatch: {name}")
                setattr(self, name, value)
        self.truth, self.cvt, self.xyz = self.truth_gev, self.cvt_gev, self.xyz_cm
        sizes = self.meta["role_sizes"]
        if set(sizes) != {"train", "validation", "evaluation"} or any(type(v) is not int or v < 0 for v in sizes.values()):
            raise ValueError("Invalid role sizes")
        n = sum(sizes.values())
        if (self.offsets.dtype.kind not in "iu" or self.offsets.shape != (n + 1,) or self.offsets[0] != 0
                or self.truth.shape != (n, 3) or self.cvt.shape != (n, 3)
                or any(v.dtype.kind != "f" for v in (self.truth, self.cvt, self.xyz))
                or self.xyz.shape != (self.offsets[-1], 3) or self.point_index.shape != (len(self.xyz),)
                or not np.isfinite(self.xyz).all() or not np.isfinite(self.truth).all()
                or not (np.isfinite(self.cvt).all(axis=1) | np.isnan(self.cvt).all(axis=1)).all()
                or not np.all((np.diff(self.offsets) >= self.meta["min_clusters"])
                              & (np.diff(self.offsets) <= self.meta["max_clusters"]))):
            raise ValueError("Invalid packed track arrays")
        for name in (*IDENTITY_FIELDS, "segment_label", "source_event_index"):
            value = getattr(self, name)
            if (value.shape != (n,) or (name == "source_manifest_id" and value.dtype.kind != "U")
                    or (name != "source_manifest_id" and (value.dtype.kind not in "iu" or np.any(value < 0)))):
                raise ValueError(f"Invalid packed identity: {name}")
        if self.point_index.dtype.kind not in "iu" or np.any(self.point_index < 0) or np.any(self.primary_tid <= 0):
            raise ValueError("Invalid point indices/primary tids")
        self.bounds = np.r_[0, np.cumsum([sizes[k] for k in ("train", "validation", "evaluation")])]
        self.keys = np.column_stack((self.source_event_index, self.segment_label))
        budgets = self.meta.get("train_track_budgets", [])
        if (any(type(b) is not int or not 1 <= b <= sizes["train"] for b in budgets)
                or len(set(budgets)) != len(budgets)):
            raise ValueError("Invalid frozen training track budgets")
        self.available_roles = ["train", "validation", "evaluation", *(f"train_{b}" for b in self.meta["train_track_budgets"])]
        seen_tracks, seen_events = set(), {}
        for role in ("train", "validation", "evaluation"):
            for i in self.rows(role):
                key = self.identity(i)
                if not key[0] or key in seen_tracks or seen_events.setdefault(key[:-1], role) != role:
                    raise ValueError("Duplicate or overlapping physical momentum identity")
                seen_tracks.add(key)
                indices = self.hits(i)[1]
                if np.any(indices[1:] <= indices[:-1]):
                    raise ValueError("Original point indices must be unique and ordered")

    def rows(self, role):
        if role in {"train", "validation", "evaluation"}:
            i = ("train", "validation", "evaluation").index(role)
            return np.arange(self.bounds[i], self.bounds[i + 1])
        if role.startswith("train_") and role in self.available_roles:
            return np.arange(int(role.removeprefix("train_")))
        raise ValueError(f"Unsupported role: {role}")

    def identity(self, i):
        return str(self.source_manifest_id[i]), int(self.source_id[i]), int(self.original_ordinal[i]), int(self.primary_tid[i])

    def hits(self, i):
        start, stop = self.offsets[i:i + 2]
        return self.xyz[start:stop], self.point_index[start:stop]
