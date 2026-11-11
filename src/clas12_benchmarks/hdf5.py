"""Read-only, schema-checked access to published CVT cluster shards."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
import operator
import os
from pathlib import Path, PurePosixPath

import numpy as np

FORMAT = "cvt-clusters-h5-v1"
COHORT_SCHEMA = "clas12_baseline_cohort_v2"
NOISE_POLICY = "primary_unassigned_is_noise_v1"
NATIVE_NOISE_POLICY = "primary_valid_only_v1"
MIN4_NOISE_POLICY = "primary_unassigned_or_lt4_is_noise_v2"
NOISE_POLICIES = (NOISE_POLICY, NATIVE_NOISE_POLICY, MIN4_NOISE_POLICY)
PID_POLICY = "abs_pdg_pion_kaon_proton_electron_other_v1"
FEATURE_RECIPE = "cluster_xyz_eta_phi_radius_v1"
POINT_FIELDS = ("clusters.xyz", "clusters.primary_segment", "clusters.reco_segment",
                "clusters.primary_tid", "clusters.primary_pid", "clusters.primary_pid_valid",
                "clusters.primary_noise", "clusters.primary_noise_valid", "particles.tid")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_digest(digest, field, value):
    value = np.asarray(value)
    digest.update(json.dumps([field, value.shape, value.dtype.descr if value.dtype.names
                              else value.dtype.str], sort_keys=True).encode())
    digest.update(value.tobytes(order="C"))


class ClusterSource:
    """Manifest-order events; each ragged domain uses its own offsets.

    h5py is intentionally imported only when this reader opens a part. Metadata
    fingerprints identify the publication; content digests identify selected data.
    """

    def __init__(self, root, split, *, max_open=4):
        self.root = Path(root).resolve()
        self.split = split
        self.max_open = operator.index(max_open)
        if self.max_open < 1:
            raise ValueError("max_open must be positive")
        self.manifest = json.loads(self._path("manifest.json").read_text())
        self.schema = json.loads(self._path("schema.json").read_text())
        if (self.manifest.get("format") != FORMAT or self.manifest.get("status") != "complete"
                or self.schema.get("format") != FORMAT or not self.manifest.get("plan_id")
                or not self.manifest.get("source_manifest_id")):
            raise ValueError(f"Expected complete {FORMAT} publication: {self.root}")
        required = {"events": {"source_id": ("int32", []), "original_ordinal": ("int64", []),
                                "run": ("int32", []), "event": ("int32", [])},
                    "clusters": {"xyz": ("float32", [3]), "primary_noise": ("int8", []),
                                 **{key: ("int32", []) for key in
                                    ("primary_segment", "reco_segment", "primary_tid", "primary_pid")},
                                 **{key: ("bool", []) for key in ("primary_pid_valid", "primary_noise_valid")}},
                    "particles": {"tid": ("int32", []), "observed_momentum": ("float32", [3]),
                                  "observed_valid": ("bool", [3])}}
        for domain, columns in required.items():
            for key, (dtype, shape) in columns.items():
                field = self.schema.get("fields", {}).get(domain, {}).get(key)
                if field is None or np.dtype(field["dtype"]) != np.dtype(dtype) or field["row_shape"] != shape:
                    raise ValueError(f"Unsupported {FORMAT} schema field: {domain}/{key}")
        if split not in {"train", "validation", "test"} or split not in self.manifest["splits"]:
            raise ValueError(f"Unsupported or absent source split: {split}")
        self.sources = {}
        for row in self.manifest["sources"]:
            sid = row["id"]
            if (type(sid) is not int or sid < 0 or sid in self.sources
                    or row["split"] not in self.manifest["splits"]):
                raise ValueError("Invalid/duplicate manifest source ID")
            self.sources[sid] = row
        self.parts, self.ends = [], []
        seen, total = set(), 0
        for part in self.manifest["splits"][split]:
            path = PurePosixPath(part["path"])
            count = part["events"]
            if (path.is_absolute() or path.parent != PurePosixPath(split) or ".." in path.parts
                    or path.suffix != ".h5" or str(path) in seen or part["split"] != split
                    or type(count) is not int or count < 0):
                raise ValueError(f"Invalid manifest part: {part}")
            self._path(str(path))
            seen.add(str(path))
            self.parts.append((str(path), count))
            total += count
            self.ends.append(total)
        self.fingerprints = {name: _sha(self._path(name)) for name in
                             ("manifest.json", "schema.json", "recipe.json")}
        geometry = self.root / "geometry"
        if not geometry.is_dir():
            raise FileNotFoundError(geometry)
        for path in sorted(geometry.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.root).as_posix()
                self.fingerprints[relative] = _sha(self._path(relative))
        self._files = OrderedDict()
        self._validated = {}
        self._pid = os.getpid()

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Source path escapes dataset root: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    @property
    def contract(self):
        return {"format": FORMAT, "token_kind": "clusters", "split": self.split,
                "plan_id": self.manifest["plan_id"],
                "source_manifest_id": self.manifest["source_manifest_id"],
                "fingerprints": self.fingerprints}

    def _file(self, part):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if part not in self._files:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("Cluster HDF5 reading requires the 'hdf5' extra") from exc
            name, count = self.parts[part]
            file = h5py.File(self._path(name), "r")
            try:
                if (file.attrs.get("format") != FORMAT or file.attrs.get("complete") != 1
                        or file.attrs.get("plan_id") != self.manifest["plan_id"]
                        or file.attrs.get("part_key") != name or file.attrs.get("events") != count):
                    raise ValueError(f"Part attributes disagree with manifest: {name}")
            except BaseException:
                file.close()
                raise
            self._files[part] = file
            self._validated[part] = set()
            if len(self._files) > self.max_open:
                old, handle = self._files.popitem(last=False)
                handle.close()
                self._validated.pop(old, None)
        self._files.move_to_end(part)
        return self._files[part]

    def _domain(self, part, domain):
        file = self._file(part)
        if domain in self._validated[part]:
            return file
        if domain not in self.schema["fields"]:
            raise ValueError(f"Unknown schema domain: {domain}")
        count = self.parts[part][1]
        try:
            if domain == "events":
                total = count
            else:
                offsets = file[domain + "/offsets"]
                if offsets.dtype != np.dtype("int64") or offsets.shape != (count + 1,):
                    raise ValueError("Offsets must have int64 event-count-plus-one shape")
                values = offsets[:]
                if values[0] != 0 or np.any(values[1:] < values[:-1]):
                    raise ValueError("Offsets must start at zero and be monotonic")
                total = int(values[-1])
            for key, spec in self.schema["fields"][domain].items():
                column = file[domain + "/" + key]
                dtype = spec["dtype"]
                dtype = np.dtype([tuple(v) for v in dtype]) if isinstance(dtype, list) else np.dtype(dtype)
                if column.shape != (total, *spec["row_shape"]) or column.dtype != dtype:
                    raise ValueError(f"Field shape/dtype mismatch: {domain}/{key}")
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid {domain} in {self.parts[part][0]}: {exc}") from exc
        self._validated[part].add(domain)
        return file

    def locate(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        return part, index - (self.ends[part - 1] if part else 0)

    def read(self, index, fields):
        part, local = self.locate(index)
        result = {}
        for field in fields:
            path = field.replace(".", "/")
            domain, key = path.rsplit("/", 1)
            if key not in self.schema["fields"].get(domain, {}):
                raise ValueError(f"Unknown event field: {field}")
            file = self._domain(part, domain)
            if domain == "events":
                result[field] = file[path][local]
            else:
                start, stop = map(int, file[domain + "/offsets"][local:local + 2])
                result[field] = file[path][start:stop]
        return result

    def validate_part(self, part):
        """Validate every declared domain, including complete offsets, for audits."""
        for domain in self.schema["fields"]:
            self._domain(part, domain)

    def identities(self, indices):
        """Gather scalar identities per shard, rather than one read per event."""
        grouped, result = {}, {}
        for index in sorted(set(indices)):
            part, local = self.locate(index)
            grouped.setdefault(part, []).append((index, local))
        for part, items in grouped.items():
            file = self._domain(part, "events")
            local = sorted({row for _, row in items})
            position = {row: j for j, row in enumerate(local)}
            values = {key: file["events/" + key][local] for key in
                      ("source_id", "original_ordinal", "run", "event")}
            for index, local_row in items:
                j = position[local_row]
                sid, ordinal = int(values["source_id"][j]), int(values["original_ordinal"][j])
                source = self.sources.get(sid)
                if (source is None or source["split"] != self.split or ordinal < 0
                        or ordinal >= source["count"]):
                    raise ValueError(f"Invalid source identity at {self.split}:{index}")
                result[index] = {"source_manifest_id": self.manifest["source_manifest_id"],
                                 "source_id": sid, "original_ordinal": ordinal,
                                 "native_run": int(values["run"][j]),
                                 "native_event": int(values["event"][j])}
        return result

    def overlap_key(self, identity):
        source = self.sources[identity["source_id"]]
        fingerprint = source.get("signal", {}).get("sha256")
        if not fingerprint:
            raise ValueError("Source inventory needs an original signal content hash")
        return fingerprint, identity["original_ordinal"]

    def content_digest(self, indices, fields=POINT_FIELDS):
        indices = sorted(set(indices))
        current = {name: _sha(self._path(name)) for name in self.fingerprints}
        geometry_files = {path.relative_to(self.root).as_posix() for path in
                          (self.root / "geometry").rglob("*") if path.is_file()}
        expected_geometry = {name for name in self.fingerprints if name.startswith("geometry/")}
        if current != self.fingerprints or geometry_files != expected_geometry:
            raise ValueError("Source metadata/geometry changed since opening")
        digest = hashlib.sha256()
        identities = self.identities(indices)
        for index in indices:
            digest.update(json.dumps([index, identities[index]], sort_keys=True).encode())
            for name, array in self.read(index, fields).items():
                update_digest(digest, name, array)
        return digest.hexdigest()

    def close(self):
        for file in self._files.values():
            file.close()
        self._files.clear()
        self._validated.clear()

    def __getstate__(self):
        return {**self.__dict__, "_files": OrderedDict(), "_validated": {}}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def cluster_projection(rows, *, noise_policy=NOISE_POLICY, pid_policy=PID_POLICY):
    """Project labels only; measurement features are built independently."""
    seg = np.asarray(rows["clusters.primary_segment"], dtype=np.int64)
    reco = np.asarray(rows["clusters.reco_segment"], dtype=np.int64)
    n = len(rows["clusters.xyz"])
    if (seg.shape != (n,) or reco.shape != (n,) or np.any(seg < -1) or np.any(reco < -1)):
        raise ValueError("Invalid cluster segmentation alignment or labels")
    owners, tids = rows["clusters.primary_tid"], rows["particles.tid"]
    if (owners.shape != (n,) or np.any(tids <= 0) or len(np.unique(tids)) != len(tids)
            or np.any((owners != -1) & (owners <= 0))
            or np.any((seg < 0) != (owners < 0)) or not np.isin(owners[owners > 0], tids).all()):
        raise ValueError("Invalid cluster owner/particle association")
    used = set()
    for label in np.unique(seg[seg >= 0]):
        owner = np.unique(owners[seg == label])
        if len(owner) != 1 or owner[0] <= 0 or int(owner[0]) in used:
            raise ValueError("Conflicting segment/particle association")
        used.add(int(owner[0]))
    raw_noise = np.asarray(rows["clusters.primary_noise"])
    noise_valid = np.asarray(rows["clusters.primary_noise_valid"], dtype=bool)
    pid = np.asarray(rows["clusters.primary_pid"])
    pid_valid = np.asarray(rows["clusters.primary_pid_valid"], dtype=bool)
    if any(v.shape != (n,) for v in (raw_noise, noise_valid, pid, pid_valid)):
        raise ValueError("Invalid cluster target alignment")
    if np.any(~np.isin(raw_noise, [-1, 0, 1])) or np.any(noise_valid & (raw_noise < 0)):
        raise ValueError("Invalid native noise label/validity")
    if noise_policy == NOISE_POLICY:
        noise, mask = (seg < 0).astype(np.int64), np.ones(n, dtype=bool)
    elif noise_policy == MIN4_NOISE_POLICY:
        from .fm_reference.track_eligibility import eligible_primary_labels

        noise = (eligible_primary_labels(seg) < 0).astype(np.int64)
        mask = np.ones(n, dtype=bool)
    elif noise_policy == NATIVE_NOISE_POLICY:
        mask = noise_valid
        noise = np.where(mask, raw_noise, -100).astype(np.int64)
    else:
        raise ValueError(f"Unknown noise policy: {noise_policy}")
    if pid_policy != PID_POLICY:
        raise ValueError(f"Unknown PID policy: {pid_policy}")
    mapped = np.full(n, 4, dtype=np.int64)
    for code, label in ((211, 0), (321, 1), (2212, 2), (11, 3)):
        mapped[np.abs(pid.astype(np.int64)) == code] = label
    mapped[~pid_valid] = -100
    labels = {"seg_target": seg, "coatjava_seg_pred": reco,
              "noise_target": noise, "pid_target": mapped}
    masks = {"noise_target": mask, "pid_target": pid_valid,
             "native_noise": noise_valid}
    return labels, masks
