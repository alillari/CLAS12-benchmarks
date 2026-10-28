"""Explicit event cohorts; no dependency on CLAS12's models or trainers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Event:
    identity: dict
    raw_features: np.ndarray
    positions: np.ndarray  # eta, cos(phi), sin(phi), radius, explicitly normalized
    energy: np.ndarray | None
    labels: dict[str, np.ndarray]
    original_point_index: np.ndarray


class Cohort:
    """Read only explicit indices, preserve all points, and reject event leakage."""

    def __init__(self, manifest):
        from mmap_ninja import RaggedMmap

        self.path = Path(manifest).resolve()
        self.spec = json.loads(self.path.read_text())
        if self.spec.get("schema") != "clas12_baseline_cohort_v1":
            raise ValueError("Unsupported cohort schema")
        self.sha256 = file_sha256(self.path)
        self.roles = self.spec["roles"]
        if not self.roles or set(self.roles) - {"train", "val", "test"}:
            raise ValueError("Roles must be train, val, or test")
        requested = {name: set() for name in self.spec["datasets"]}
        for members in self.roles.values():
            for member in members:
                name, index = member["dataset"], member["index"]
                if type(index) is not int or index < 0:
                    raise ValueError(f"Invalid event index: {member}")
                requested[name].add(index)
        self.datasets = {}
        for name, spec in self.spec["datasets"].items():
            root = Path(spec["root"])
            if not root.is_absolute():
                root = self.path.parent / root
            split = spec["source_split"]
            if spec["input_layout"] not in {"xyz", "exyz"}:
                raise ValueError("input_layout must explicitly be xyz or exyz")
            for key in ("eta_scale", "radius_scale"):
                if not np.isfinite(spec[key]) or spec[key] <= 0:
                    raise ValueError(f"{key} must be positive and finite")
            metadata = root / "metadata.json"
            if file_sha256(metadata) != spec["metadata_sha256"]:
                raise ValueError(f"Dataset metadata hash mismatch: {name}")
            samples = root / f"samples_{split}.jsonl"
            if file_sha256(samples) != spec["samples_sha256"]:
                raise ValueError(f"Sample identities hash mismatch: {name}")
            rows, total_rows = {}, 0
            with samples.open() as stream:
                for index, line in enumerate(stream):
                    if index in requested[name]:
                        rows[index] = json.loads(line)
                    total_rows += 1
            if set(rows) != requested[name]:
                raise ValueError(f"Invalid event index in dataset {name}")
            arrays = {}
            for field in ["features", "seg_target", *spec.get("extra_targets", [])]:
                if field in arrays or field not in {"features", "seg_target", "pid_target", "noise_target"}:
                    raise ValueError(f"Duplicate or unsupported target: {field}")
                location = root / f"{field}_{split}"
                # RaggedMmap's constructor creates missing directories; check first.
                if not location.is_dir():
                    raise FileNotFoundError(location)
                arrays[field] = RaggedMmap(location, mode="r")
                if len(arrays[field]) != total_rows:
                    raise ValueError(f"Event count mismatch for {name}/{field}")
            self.datasets[name] = (spec, rows, arrays)
        seen = {}
        for role, members in self.roles.items():
            for member in members:
                name, index = member["dataset"], member["index"]
                _, rows, _ = self.datasets[name]
                if index not in rows:
                    raise ValueError(f"Invalid event index: {member}")
                row = rows[index]
                if row.get("sample_mode") != "event":
                    raise ValueError("Baseline cohorts require full event samples")
                if not row.get("source_file") or type(row.get("event")) is not int:
                    raise ValueError(f"Missing source-file/event identity: {member}")
                identity = (row["source_file"], row["event"])
                if identity in seen:
                    raise ValueError(f"Duplicate source event in {seen[identity]} and {role}: {identity}")
                seen[identity] = role

    def event(self, role, number):
        member = self.roles[role][number]
        name, index = member["dataset"], member["index"]
        spec, rows, arrays = self.datasets[name]
        raw = np.array(arrays["features"][index], dtype=np.float32, copy=True)
        width = 3 if spec["input_layout"] == "xyz" else 4
        if raw.ndim != 2 or raw.shape[1] != width or not np.isfinite(raw).all():
            raise ValueError(f"Invalid feature shape or nonfinite values: {name}:{index}")
        xyz = raw[:, -3:]
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        if np.any(radius <= 0):
            raise ValueError("Zero transverse radius cannot define eta/phi")
        eta = np.arcsinh(xyz[:, 2] / radius) / spec["eta_scale"]
        positions = np.column_stack((eta, xyz[:, 0] / radius, xyz[:, 1] / radius,
                                     radius / spec["radius_scale"])).astype(np.float32)
        labels = {}
        for field, array in arrays.items():
            if field == "features":
                continue
            values = np.array(array[index], copy=True)
            if values.shape != (len(raw),) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"Point alignment or dtype mismatch: {name}:{index}/{field}")
            labels[field] = values.astype(np.int64)
        if np.any(labels["seg_target"] < -1):
            raise ValueError("seg_target must use -1 for background, >=0 for signal")
        row = rows[index]
        identity = {"source_file": row["source_file"], "event": row["event"],
                    "dataset": name, "source_split": spec["source_split"], "dataset_index": index}
        return Event(identity, raw, positions, raw[:, 0] if width == 4 else None,
                     labels, np.arange(len(raw), dtype=np.int64))


def node_features(event, *, allow_position_only=False):
    if event.energy is None:
        if not allow_position_only:
            raise ValueError("Published input includes energy. Choose an explicit position-only adaptation; no dummy energy is inserted.")
        return event.positions.copy()
    return np.column_stack((event.energy, event.positions)).astype(np.float32)


def eggnet_features(event):
    """Self plus two distinct nearest neighbors; never consult labels."""
    from scipy.spatial import cKDTree

    x = event.positions
    if len(x) < 3:
        raise ValueError("EggNet's 12D input requires at least three points")
    _, indices = cKDTree(x).query(x, k=3)
    neighbors = []
    for i, candidates in enumerate(indices):
        candidates = candidates[candidates != i][:2]
        if len(candidates) != 2:
            raise ValueError("Cannot construct two distinct neighbors")
        neighbors.append(candidates)
    return np.concatenate((x, x[np.asarray(neighbors)].reshape(len(x), 8)), axis=1)


def classification_edges(positions, k=50, radius=1.0):
    """Directed neighbor -> center edges, excluding self; strict distance cut."""
    from scipy.spatial import cKDTree

    if k <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("k and radius must be positive")
    n = len(positions)
    if n < 2:
        return np.empty((2, 0), dtype=np.int64)
    distances, indices = cKDTree(positions).query(positions, k=min(k + 1, n))
    edges = []
    for center, (ds, ns) in enumerate(zip(distances, indices)):
        selected = ns[(ns != center) & (ds < radius)][:k]
        edges.extend((int(neighbor), center) for neighbor in selected)
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T


def positive_edges(labels, max_edges=1_000_000):
    """Training-only all-pairs same-track supervision; background is excluded."""
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels) if label >= 0]
    count = sum(len(group) * (len(group) - 1) for group in groups)
    if count > max_edges:
        raise ValueError(f"{count} truth edges exceed explicit limit {max_edges}; no truncation performed")
    edges = []
    for group in groups:
        a, b = np.meshgrid(group, group, indexing="ij")
        keep = a != b
        edges.append(np.stack((a[keep], b[keep])))
    return np.concatenate(edges, axis=1) if edges else np.empty((2, 0), dtype=np.int64)
