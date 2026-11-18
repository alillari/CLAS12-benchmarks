"""Build deterministic track cohorts from bounded windows across every strip shard."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

LAYERS = 12
RAW_LENGTH = 48
FEATURES = 15
TRACK_FIELDS = ("truth", "coatjava", "identity", "pid", "coverage", "count", "layer",
                "raw", "raw_layer")


def ragged(group, lo, hi, fields):
    offsets = group["offsets"][lo:hi + 1]
    first, last = int(offsets[0]), int(offsets[-1])
    result = {name: group[name][first:last] for name in fields}
    return result, np.repeat(np.arange(hi - lo), np.diff(offsets))


def keys(events, tids):
    return (events.astype(np.int64) << 32) | tids.astype(np.uint32)


def lookup(sorted_keys, order, wanted):
    indices = np.searchsorted(sorted_keys, wanted)
    indices = np.minimum(indices, len(sorted_keys) - 1)
    return np.where(sorted_keys[indices] == wanted, order[indices], -1)


def unpack(file, lo, hi, compare, save_raw):
    p, pe = ragged(file["particles"], lo, hi,
                  ("tid", "observed_momentum", "observed_valid", "pid", "coverage"))
    s, se = ragged(file["strips"], lo, hi,
                  ("primary_tid", "detector", "layer", "xyz", "endpoints", "arc_angle",
                   "pitch", "energy", "adc", "time", "raw_time"))
    valid = s["primary_tid"] > 0
    tracks = keys(pe, p["tid"])
    order = tracks.argsort()
    ordered = tracks[order]
    assert len(ordered) and np.all(np.diff(ordered) > 0)
    owner = lookup(ordered, order, keys(se[valid], s["primary_tid"][valid]))
    if (owner < 0).any():
        raise ValueError("Direct primary strip lacks an event-local particle")
    s = {k: v[valid] for k, v in s.items()}
    layer = s["layer"] - 1 + 6 * (s["detector"] == 1)
    assert (np.isin(s["detector"], (1, 2)) & (s["layer"] >= 1) & (s["layer"] <= 6)).all()
    counts = np.bincount(owner * LAYERS + layer, minlength=len(tracks) * LAYERS).reshape(-1, LAYERS)
    count = counts.sum(1)
    y = p["observed_momentum"]
    accept = (count >= 4) & ((counts > 0).sum(1) >= 4) & p["observed_valid"].all(1)
    accept &= np.isfinite(y).all(1) & (np.linalg.norm(y, axis=1) > 0)
    indices = np.flatnonzero(accept)
    local = np.cumsum(accept) - 1
    hit = accept[owner]
    track = local[owner[hit]]
    layer = layer[hit]
    s = {k: v[hit] for k, v in s.items()}

    raw = np.column_stack((
        s["xyz"] / 30, s["endpoints"] / 30, s["arc_angle"],
        np.log1p(s["pitch"] * 100), np.log1p(np.maximum(s["energy"], 0)),
        np.log1p(np.maximum(s["adc"], 0)), np.arcsinh(s["time"] / 100),
        np.arcsinh(s["raw_time"] / 100),
    )).astype(np.float32)
    if not np.isfinite(raw[:, :11]).all():
        raise ValueError("Nonfinite strip geometry")
    raw[:, 11:] = np.nan_to_num(raw[:, 11:], nan=0, posinf=0, neginf=0)

    n = len(indices)
    shape = (n, LAYERS)
    group = track * LAYERS + layer
    flat_count = counts[accept].reshape(-1)
    weight = np.sqrt(np.maximum(s["adc"], 1)).astype(np.float32)
    denom = np.bincount(group, weights=weight, minlength=n * LAYERS)
    geom = np.column_stack([
        np.bincount(group, weights=(values * weight if j < 11 else values), minlength=n * LAYERS)
        / np.maximum(denom if j < 11 else flat_count, 1)
        for j, values in enumerate(raw.T)
    ]).reshape(*shape, FEATURES)
    unweighted = np.column_stack([
        np.bincount(group, weights=values, minlength=n * LAYERS) / np.maximum(flat_count, 1)
        for values in raw[:, :3].T
    ]).reshape(*shape, 3)
    geom = np.concatenate((geom, unweighted, np.log1p(counts[accept])[..., None]), axis=-1)

    sequence = {}
    if save_raw:
        ordering = np.argsort(track, kind="stable")
        ordered_tracks = track[ordering]
        beginnings = np.cumsum(count[accept]) - count[accept]
        within = np.arange(len(ordering)) - beginnings[ordered_tracks]
        retain = within < RAW_LENGTH
        retained = ordering[retain]
        raw_x = np.zeros((n, RAW_LENGTH, FEATURES), np.float16)
        raw_layer = np.full((n, RAW_LENGTH), -1, np.int8)
        raw_x[track[retained], within[retain]] = raw[retained]
        raw_layer[track[retained], within[retain]] = layer[retained]
        sequence = dict(raw=raw_x, raw_layer=raw_layer)

    benchmark = np.full((len(tracks), 3), np.nan, np.float32)
    if compare:
        r, re = ragged(file["reco/cvt_tracks"], lo, hi, (
            "purity_tid", "purity_valid", "unique_owner", "benchmark_p", "benchmark_theta",
            "entrance_phi"))
        eligible = (r["purity_tid"] > 0) & r["purity_valid"] & r["unique_owner"]
        match = lookup(ordered, order, keys(re[eligible], r["purity_tid"][eligible]))
        p_cvt, theta, phi = (r[name][eligible] for name in
                            ("benchmark_p", "benchmark_theta", "entrance_phi"))
        multiplicity = np.bincount(match[match >= 0], minlength=len(tracks))
        good = (match >= 0) & (multiplicity[match] == 1) & (p_cvt > 0)
        good &= np.isfinite(p_cvt + theta + phi) & (theta >= 0) & (theta <= np.pi)
        vec = p_cvt[:, None] * np.column_stack(
            (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
        benchmark[match[good]] = vec[good]
    event_id = np.column_stack([file["events/" + key][lo:hi] for key in
                                ("source_id", "original_ordinal")])
    identity = np.column_stack((event_id[pe[indices]], p["tid"][indices]))
    return dict(truth=y[indices], coatjava=benchmark[indices], identity=identity,
                pid=p["pid"][indices], coverage=p["coverage"][indices],
                count=count[indices].astype(np.int32), layer=geom.astype(np.float32),
                **sequence)


def read_part(task):
    path, events, per_part, seed, compare, save_raw = task
    rng = np.random.default_rng(seed)
    blocks = np.linspace(0, events, 5, dtype=int)
    width = min((per_part + 3) // 4, np.min(np.diff(blocks)))
    with h5py.File(path, "r") as f:
        outputs = []
        for start, stop in zip(blocks[:-1], blocks[1:]):
            lo = int(rng.integers(start, stop - width + 1))
            outputs.append(unpack(f, lo, lo + width, compare, save_raw))
    return {key: np.concatenate([part[key] for part in outputs]) for key in outputs[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data"))
    parser.add_argument("--out", type=Path, default=Path("/results"))
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-parts", type=int)
    parser.add_argument("--layer-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    blob = (args.root / "manifest.json").read_bytes()
    manifest = json.loads(blob)
    if manifest["status"] != "complete" or manifest["format"] != "cvt-strips-h5-v1":
        raise ValueError("Expected the complete strip v1 product")
    parts = manifest["splits"][args.split][:args.max_parts]
    sources = [{s["id"] for part in records for s in part["sources"]}
               for records in manifest["splits"].values()]
    assert all(not (first & second) for i, first in enumerate(sources) for second in sources[i + 1:])
    per_part = (args.events + len(parts) - 1) // len(parts)
    seed = {"train": 59611, "validation": 77343, "test": 31877}[args.split]
    output_parts = []
    tasks = [(args.root / part["path"], part["events"], per_part, seed + i,
              args.split != "train", not args.layer_only) for i, part in enumerate(parts)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(read_part, tasks), 1):
            output_parts.append(result)
            if i % 10 == 0 or i == len(tasks):
                print(f"{args.split}: {i}/{len(tasks)} shards", flush=True)
    fields = TRACK_FIELDS[:-2] if args.layer_only else TRACK_FIELDS
    data = {key: np.concatenate([part.pop(key) for part in output_parts]) for key in fields}
    n = len(data["truth"])
    permutation = np.random.default_rng(seed).permutation(n)
    for key in fields:
        np.save(args.out / f"{args.split}_{key}.npy", data.pop(key)[permutation])
    ids = np.load(args.out / f"{args.split}_identity.npy")
    count = np.load(args.out / f"{args.split}_count.npy")
    assert set(ids[:, 0]).issubset(sources[list(manifest["splits"]).index(args.split)])
    assert len(np.unique(ids, axis=0)) == n
    report = {
        "manifest_sha256": hashlib.sha256(blob).hexdigest(), "plan_id": manifest["plan_id"],
        "split": args.split, "shards": len(parts), "source_count": len(np.unique(ids[:, 0])),
        "tracks": n, "track_cut": "valid observed p>0, >=4 owned direct-primary strips in >=4 layers",
        "counts_percentiles": np.quantile(count, [0, .1, .5, .9, .99, 1]).tolist(),
        "raw_sequence_cap": None if args.layer_only else RAW_LENGTH,
        "raw_would_truncate": None if args.layer_only else int((count > RAW_LENGTH).sum()),
        "identity_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "sampling_seed": seed, "events_per_quartile_per_shard": (per_part + 3) // 4,
    }
    (args.out / f"{args.split}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
