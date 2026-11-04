"""Portable truth-associated CVT track manifests built from frozen companion keys."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .data import file_sha256


SCHEMA = "clas12_momentum_tracks_v1"
ROLES = ("train", "validation", "evaluation")
TRAIN_BUDGETS = ("train_1000", "train_50000")
TRUTH_COLUMNS = tuple(f"mc_entrance_{axis}" for axis in ("px", "py", "pz"))
CVT_COLUMNS = ("cvt_benchmark_p", "cvt_benchmark_theta", "cvttraj_entrance_phi")


def _check_key_hashes(keys, lengths, role, info):
    if len(keys) != info["count"]:
        raise ValueError(f"Frozen track count mismatch in {role}")
    for array, name in ((keys, "keys"), (lengths, "lengths")):
        if hashlib.sha256(array.tobytes()).hexdigest() != info[name + "_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch in {role}")


def _columns(metadata, field, names):
    layout = metadata.get(field)
    if not isinstance(layout, list) or len(set(layout)) != len(layout) or not set(names) <= set(layout):
        raise ValueError(f"Missing or ambiguous {field} for {names}")
    return len(layout), [layout.index(name) for name in names]


def _keys(prepared, role, info):
    with np.load(prepared / f"momentum_{role}.npz", allow_pickle=False) as archive:
        keys, lengths = archive["keys"], archive["lengths"]
    if (keys.shape != (info["count"], 2) or lengths.shape != (len(keys),) or not len(keys)
            or any(a.dtype != np.dtype("int64") for a in (keys, lengths))
            or (keys < 0).any() or (lengths < 12).any() or (lengths > 100).any()):
        raise ValueError(f"Invalid frozen keys or lengths in {role}")
    _check_key_hashes(keys, lengths, role, info)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError(f"Duplicate frozen track key in {role}")
    return keys, lengths


def _identities(root, split, requested):
    """Stream only up to the last selected row of the multi-GB identity file."""
    result = {}
    with (root / f"samples_{split}.jsonl").open() as stream:
        for row, line in enumerate(stream):
            if row not in requested:
                continue
            sample = json.loads(line)
            source, event = sample.get("source_file"), sample.get("event")
            if (sample.get("sample_mode") != "event" or not isinstance(source, str)
                    or not source or type(event) is not int):
                raise ValueError(f"Invalid event identity at {split}:{row}")
            result[row] = (source, event)
            if len(result) == len(requested):
                break
    if len(result) != len(requested):
        raise ValueError(f"Missing selected source event in {split}")
    return result


def _event_arrays(root, split):
    from mmap_ninja import RaggedMmap
    arrays = []
    for field in ("features", "seg_target", "reg_target", "aux_target"):
        path = root / f"{field}_{split}"
        if not path.is_dir():
            raise FileNotFoundError(path)
        arrays.append(RaggedMmap(str(path), mode="r"))
    if len({len(a) for a in arrays}) != 1:
        raise ValueError(f"Misaligned RaggedMmap event counts in {split}")
    return arrays


def _cvt_vector(aux):
    """Match the companion's first jointly finite row within this truth segment."""
    finite = np.flatnonzero(np.isfinite(aux).all(axis=1))
    if len(finite):
        p, theta, phi = aux[finite[0]].astype(np.float64)
        if p > 0 and 0 <= theta <= np.pi:
            pt = p * np.sin(theta)
            return [pt * np.cos(phi), pt * np.sin(phi), p * np.cos(theta)]
    return [np.nan] * 3


def _assemble_role(root, role, keys, lengths, info, layouts, seen_events, output):
    split = "pretrain" if role == "train" else "test"
    ids = _identities(root, split, set(keys[:, 0]))
    for row, identity in ids.items():
        if identity in seen_events:
            raise ValueError(f"Repeated source event {identity} at {role}:{row}, first in {seen_events[identity]}")
        seen_events[identity] = (role, row)

    arrays = _event_arrays(root, split)
    reg_width, reg_cols, aux_width, aux_cols = layouts
    digest = hashlib.sha256()
    hashed = set()
    last_row = None
    for (row, label), length in zip(keys, lengths, strict=True):
        if row != last_row:
            xyz, seg, reg, aux = (np.asarray(a[int(row)]) for a in arrays)
            last_row = row
            n = len(xyz)
            if (xyz.shape != (n, 3) or xyz.dtype.kind != "f" or not np.isfinite(xyz).all()
                    or seg.shape != (n,) or seg.dtype.kind not in "iu" or np.any(seg < -1)
                    or reg.shape != (n, reg_width) or aux.shape != (n, aux_width)
                    or reg.dtype.kind != "f" or aux.dtype.kind != "f"):
                raise ValueError(f"Invalid event arrays at {split}:{row}")
            if row not in hashed:
                # The companion hashes whole events in first-occurrence order.
                for array in (xyz, seg, reg):
                    digest.update(str((array.shape, array.dtype.str)).encode())
                    digest.update(array.tobytes())
                hashed.add(row)

        indices = np.flatnonzero(seg == label)
        if len(indices) != length:
            raise ValueError(f"Track length mismatch at {split}:{row}/{label}")
        values = reg[indices][:, reg_cols].astype(np.float64)
        finite = np.isfinite(values)
        if not finite.any(axis=0).all():
            raise ValueError(f"Missing entrance truth at {split}:{row}/{label}")
        native_mev = np.where(finite, values, 0).sum(axis=0) / finite.sum(axis=0)
        source, event = ids[row]
        output["source_file"].append(source)
        output["event"].append(event)
        output["xyz_cm"].append(xyz[indices])
        output["point_index"].append(indices)
        output["truth_gev"].append(native_mev * 0.001)
        output["cvt_gev"].append(_cvt_vector(aux[indices][:, aux_cols]))
    if digest.hexdigest() != info["source_content_sha256"]:
        raise ValueError(f"Frozen raw event content hash mismatch in {role}")


def build_manifest(root, prepared, destination):
    """Import both nested training budgets, validation and evaluation in frozen order."""
    root, prepared, destination = map(Path, (root, prepared, destination))
    ready = json.loads((prepared / "data_ready.json").read_text())
    provenance = ready["cohorts"]["momentum"]
    metadata_sha = file_sha256(root / "metadata.json")
    if ready.get("complete") is not True or metadata_sha != ready["metadata_sha256"]:
        raise ValueError("Incomplete cohort preparation or wrong raw product")
    metadata = json.loads((root / "metadata.json").read_text())
    if (metadata.get("feature_units") != "cm" or metadata.get("feature_layout") != ["x", "y", "z"]
            or metadata.get("seg_target_source") != "mctrue_primary_tid"):
        raise ValueError("Expected cm cluster centers with supplied MC track membership")
    reg_width, reg_cols = _columns(metadata, "reg_target_layout", TRUTH_COLUMNS)
    aux_width, aux_cols = _columns(metadata, "aux_target_layout", CVT_COLUMNS)
    loaded = {r: _keys(prepared, r, provenance[r]) for r in (*TRAIN_BUDGETS, "validation", "evaluation")}
    full, lengths = loaded["train_50000"]
    _check_key_hashes(full, lengths, "train", provenance["train"])
    index = {tuple(key): i for i, key in enumerate(full)}
    small, small_lengths = loaded["train_1000"]
    if any(tuple(key) not in index for key in small):
        raise ValueError("1k training keys are outside the 50k cohort")
    nested = np.asarray([index[tuple(key)] for key in small])
    if not np.array_equal(lengths[nested], small_lengths):
        raise ValueError("Nested training lengths disagree")
    output = {k: [] for k in ("source_file", "event", "xyz_cm", "point_index", "truth_gev", "cvt_gev")}
    seen_events, row_keys, row_lengths, role_sizes = {}, [], [], []
    for role in ROLES:
        keys, lens = loaded["train_50000" if role == "train" else role]
        _assemble_role(root, role, keys, lens, provenance[role],
                       (reg_width, reg_cols, aux_width, aux_cols), seen_events, output)
        row_keys.append(keys)
        row_lengths.append(lens)
        role_sizes.append(len(keys))

    sources, source_index = np.unique(output.pop("source_file"), return_inverse=True)
    metadata = {"schema": SCHEMA, "raw_metadata_sha256": metadata_sha,
                "preparation_sha256": file_sha256(prepared / "data_ready.json"),
                "frozen_cohorts": provenance, "role_order": ROLES, "role_sizes": role_sizes,
                "position_unit": "cm", "momentum_unit": "GeV"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete archive without replacing an existing manifest.
    with NamedTemporaryFile(dir=destination.parent) as temp:
        np.savez_compressed(temp, metadata=json.dumps(metadata), source_files=sources,
                            source_index=source_index, keys=np.concatenate(row_keys),
                            offsets=np.r_[0, np.cumsum(np.concatenate(row_lengths))], train_1000=nested,
                            xyz_cm=np.concatenate(output.pop("xyz_cm")),
                            point_index=np.concatenate(output.pop("point_index")), **output)
        temp.flush()
        os.link(temp.name, destination)
    return metadata


class TrackManifest:
    """Role-local track order, durable identities and measured model inputs."""
    def __init__(self, path):
        self.sha256 = file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            self.meta = json.loads(str(archive["metadata"]))
            if self.meta.get("schema") != SCHEMA or self.meta["role_order"] != list(ROLES):
                raise ValueError("Unsupported momentum manifest")
            self.keys = archive["keys"]
            self.source_files = archive["source_files"]
            self.source_index = archive["source_index"]
            self.events = archive["event"]
            self.truth = archive["truth_gev"]
            self.cvt = archive["cvt_gev"]
            self.offsets = archive["offsets"]
            self.xyz = archive["xyz_cm"]
            self.point_index = archive["point_index"]
            self.train_1000 = archive["train_1000"]
        sizes = self.meta["role_sizes"]
        if len(sizes) != len(ROLES) or any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Invalid momentum role sizes")
        if self.meta.get("momentum_unit") != "GeV" or self.meta.get("position_unit") != "cm":
            raise ValueError("Unexpected momentum manifest units")
        self.bounds = np.r_[0, np.cumsum(sizes)]
        n = int(self.bounds[-1])
        if (any(array.dtype.kind not in "iu" for array in (self.keys, self.offsets, self.point_index,
                                                            self.source_index, self.events, self.train_1000))
                or self.keys.shape != (n, 2) or self.truth.shape != (n, 3) or self.cvt.shape != (n, 3)
                or self.events.shape != (n,) or self.source_index.shape != (n,)
                or self.offsets.shape != (n+1,) or self.offsets[0] != 0
                or (self.source_index < 0).any() or (self.source_index >= len(self.source_files)).any()
                or self.xyz.shape != (self.offsets[-1], 3) or self.point_index.shape != (len(self.xyz),)
                or not np.all((np.diff(self.offsets) >= 12) & (np.diff(self.offsets) <= 100))
                or not np.isfinite(self.xyz).all() or not np.isfinite(self.truth).all()
                or not (np.isfinite(self.cvt).all(axis=1) | np.isnan(self.cvt).all(axis=1)).all()):
            raise ValueError("Invalid packed track arrays")
        if (self.train_1000.ndim != 1 or len(np.unique(self.train_1000)) != len(self.train_1000)
                or (self.train_1000 < 0).any() or (self.train_1000 >= sizes[0]).any()):
            raise ValueError("Invalid nested training selection")
        lengths = np.diff(self.offsets)
        for role in (*TRAIN_BUDGETS, "validation", "evaluation"):
            rows = self.rows(role)
            _check_key_hashes(self.keys[rows], lengths[rows], role, self.meta["frozen_cohorts"][role])
        self._validate_identities()

    def _validate_identities(self):
        by_source = {}
        for role in ROLES:
            by_row = {}
            for i in self.rows(role):
                source, event, _ = self.identity(i)
                key = (source, event)
                raw_row = int(self.keys[i, 0])
                if (not source or by_source.setdefault(key, (role, raw_row)) != (role, raw_row)
                        or by_row.setdefault(raw_row, key) != key):
                    raise ValueError(f"Ambiguous or overlapping source event {key}")

    def rows(self, role):
        if role == "train_1000":
            return self.train_1000.copy()
        i = ROLES.index("train" if role == "train_50000" else role)
        return np.arange(self.bounds[i], self.bounds[i+1])

    def identity(self, i):
        return (str(self.source_files[self.source_index[i]]), int(self.events[i]), int(self.keys[i, 1]))

    def hits(self, i):
        """Return just measured xyz and their original within-event indices."""
        start, stop = self.offsets[i:i+2]
        return self.xyz[start:stop], self.point_index[start:stop]
