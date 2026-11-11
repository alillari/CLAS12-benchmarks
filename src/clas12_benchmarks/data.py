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
    masks: dict[str, np.ndarray] | None = None
    context: dict | None = None


class Cohort:
    """Read only explicit indices, preserve all points, and reject event leakage."""

    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.spec = json.loads(self.path.read_text())
        self.sources = {}
        if self.spec.get("schema") == "clas12_baseline_cohort_v2":
            self._init_hdf5()
            return
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
            from .sources import RaggedSource
            source = RaggedSource(root, split, spec, requested[name])
            self.sources[name] = source
            rows, arrays = source.rows, source.arrays
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
        if self.spec["schema"] == "clas12_baseline_cohort_v2":
            return self._hdf5_event(role, number)
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
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite normalized coordinate features")
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

    def dataset_spec(self, name):
        return self.datasets[name][0]

    @property
    def is_hdf5(self):
        return self.spec["schema"] == "clas12_baseline_cohort_v2"

    def _init_hdf5(self):
        from .hdf5 import ClusterSource, FORMAT, FEATURE_RECIPE, NOISE_POLICIES, PID_POLICY
        self.sha256 = file_sha256(self.path)
        self.roles = self.spec["roles"]
        if set(self.roles) != {"train", "val", "test"}:
            raise ValueError("HDF5 cohorts require train, val and test roles")
        self.datasets = {}
        requested = {name: set() for name in self.spec["datasets"]}
        expected_split = {"train": "train", "val": "validation", "test": "test"}
        for role, members in self.roles.items():
            for member in members:
                name, index = member["dataset"], member["index"]
                if name not in requested or type(index) is not int or index < 0:
                    raise ValueError(f"Invalid cohort member: {member}")
                if self.spec["datasets"][name]["source_split"] != expected_split[role]:
                    raise ValueError(f"Wrong published source split for {role}")
                requested[name].add(index)
        seen, contracts = {}, set()
        try:
            for name, spec in self.spec["datasets"].items():
                if (spec.get("storage_format") != FORMAT or spec.get("token_kind") != "clusters"
                        or spec.get("feature_recipe") != FEATURE_RECIPE or spec.get("input_layout") != "xyz"
                        or spec.get("noise_policy") not in NOISE_POLICIES
                        or spec.get("pid_policy") != PID_POLICY):
                    raise ValueError(f"Unsupported HDF5 feature/label contract: {name}")
                for key in ("eta_scale", "radius_scale"):
                    if not np.isfinite(spec[key]) or spec[key] <= 0:
                        raise ValueError(f"{key} must be positive and finite")
                contracts.add(tuple(spec[k] for k in ("feature_recipe", "noise_policy", "pid_policy",
                                                       "eta_scale", "radius_scale")))
                root = Path(spec["root"])
                if not root.is_absolute():
                    root = self.path.parent / root
                source = ClusterSource(root, spec["source_split"])
                self.sources[name] = source
                if source.contract != spec["source_contract"]:
                    raise ValueError(f"Source metadata/contract drift: {name}")
                rows = source.identities(requested[name])
                if source.content_digest(requested[name]) != spec["selected_content_sha256"]:
                    raise ValueError(f"Selected source content drift: {name}")
                self.datasets[name] = (spec, rows, {})
            if len(contracts) != 1:
                raise ValueError("Cannot mix feature/label/normalization contracts")
            for role, members in self.roles.items():
                for member in members:
                    name, index = member["dataset"], member["index"]
                    identity = self.datasets[name][1][index]
                    key = self.sources[name].overlap_key(identity)
                    if key in seen:
                        raise ValueError(f"Duplicate source event in {seen[key]} and {role}: {key}")
                    seen[key] = role
        except BaseException:
            self.close()
            raise

    def _hdf5_event(self, role, number):
        from .hdf5 import POINT_FIELDS, cluster_projection
        member = self.roles[role][number]
        name, index = member["dataset"], member["index"]
        spec, rows, _ = self.datasets[name]
        raw_rows = self.sources[name].read(index, POINT_FIELDS)
        raw = np.array(raw_rows["clusters.xyz"], dtype=np.float32, copy=True)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid cluster coordinates: {name}:{index}")
        radius = np.hypot(raw[:, 0], raw[:, 1])
        if np.any(radius <= 0):
            raise ValueError("Zero transverse radius cannot define eta/phi")
        positions = np.column_stack((np.arcsinh(raw[:, 2] / radius) / spec["eta_scale"],
                                     raw[:, 0] / radius, raw[:, 1] / radius,
                                     radius / spec["radius_scale"])).astype(np.float32)
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite normalized coordinate features")
        labels, masks = cluster_projection(raw_rows, noise_policy=spec["noise_policy"],
                                          pid_policy=spec["pid_policy"])
        identity = dict(rows[index], dataset=name, source_split=spec["source_split"], dataset_index=index)
        return Event(identity, raw, positions, None, labels, np.arange(len(raw), dtype=np.int64),
                     masks, {key: value for key, value in raw_rows.items() if key != "clusters.xyz"})

    def data_fingerprints(self):
        if self.is_hdf5:
            # Recheck selected content rather than trusting only the JSON receipt.
            result = {}
            for name, (spec, rows, _) in self.datasets.items():
                digest = self.sources[name].content_digest(rows)
                if digest != spec["selected_content_sha256"]:
                    raise ValueError(f"Selected source content drift: {name}")
                result[name] = {"contract": spec["source_contract"], "content_sha256": digest,
                                "feature_recipe": spec["feature_recipe"],
                                "noise_policy": spec["noise_policy"], "pid_policy": spec["pid_policy"]}
            return result
        root = self.path.parent
        roots = {}
        for name, (spec, _, _) in self.datasets.items():
            location = Path(spec["root"])
            roots[name] = (location if location.is_absolute() else root / location).resolve()
        if any(location != root for location in roots.values()):
            # A v1 manifest can reference arrays outside its own directory. The
            # historical directory walk below would otherwise hash no inputs.
            result = {}
            for name, (_, rows, arrays) in self.datasets.items():
                digest = hashlib.sha256()
                for index in sorted(rows):
                    digest.update(json.dumps([index, rows[index]], sort_keys=True).encode())
                    for field, values in arrays.items():
                        array = np.asarray(values[index])
                        digest.update(str((field, array.shape, array.dtype.str)).encode())
                        digest.update(array.tobytes())
                result[name] = {"selected_content_sha256": digest.hexdigest()}
            return result
        # Retain existing fingerprints for archived materialized v1 cohorts.
        return {str(p.relative_to(root)): file_sha256(p)
                for directory in sorted(root.iterdir()) if directory.is_dir()
                for p in sorted(directory.rglob("*")) if p.is_file()}

    def close(self):
        for source in self.sources.values():
            source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def event_key(identity):
    """Physical identity, independent of cohort aliases and row positions."""
    if "source_manifest_id" in identity:
        return (identity["source_manifest_id"], identity["source_id"], identity["original_ordinal"])
    return identity["source_file"], identity["event"]


def source_key(identity):
    return event_key(identity)[:-1]


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
