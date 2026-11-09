"""Validate all shard metadata and compare deterministic event samples to h5py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np

from clas12_benchmarks.hdf5 import ClusterSource, POINT_FIELDS, update_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=16)
    parser.add_argument("--soak-reads", type=int, default=10000)
    args = parser.parse_args()
    if args.samples_per_shard < 1 or args.soak_reads < 1 or args.output.exists():
        raise ValueError("Use positive qualification budgets and a new output filename")
    report, digest = {"splits": {}}, hashlib.sha256()
    started = time.monotonic()
    fields = [*POINT_FIELDS, "particles.observed_momentum", "particles.observed_valid"]
    for split in ("train", "validation", "test"):
        count = points = 0
        with ClusterSource(args.root, split) as source:
            for part, (name, size) in enumerate(source.parts):
                source.validate_part(part)
                indices = sorted(set([0, size - 1, *np.random.default_rng(1337 + part).choice(
                    size, min(size, args.samples_per_shard), replace=False).tolist()])) if size else []
                begin = source.ends[part - 1] if part else 0
                identities = source.identities([begin + i for i in indices])
                with h5py.File(args.root / name, "r") as reference:
                    for local in indices:
                        rows = source.read(begin + local, fields)
                        digest.update(json.dumps([split, identities[begin + local]], sort_keys=True).encode())
                        for key, actual in rows.items():
                            path = key.replace(".", "/")
                            domain = path.rsplit("/", 1)[0]
                            a, b = reference[domain + "/offsets"][local:local + 2]
                            expected = reference[path][int(a):int(b)]
                            np.testing.assert_array_equal(actual, expected)
                            update_digest(digest, key, actual)
                        count += 1; points += len(rows["clusters.xyz"])
                if (part + 1) % 50 == 0:
                    print(json.dumps({"split": split, "validated_parts": part + 1, "sampled_events": count}), flush=True)
            report["splits"][split] = {"parts": len(source.parts), "events": len(source),
                                        "sampled_events": count, "sampled_points": points}
            report["fingerprints"] = source.fingerprints
    # Fixed cache budget; force repeated eviction across eight train shards.
    descriptors = lambda: len(list(Path("/proc/self/fd").iterdir()))
    before = descriptors(); memory = []; high_handles = 0
    with ClusterSource(args.root, "train", max_open=2) as source:
        for i in range(args.soak_reads):
            part = i % min(8, len(source.parts))
            begin = source.ends[part - 1] if part else 0
            source.read(begin, ["clusters.xyz"])
            high_handles = max(high_handles, len(source._files))
            if len(source._files) > 2:
                raise ValueError("File handle cache grew beyond its budget")
            if (i + 1) % 1000 == 0:
                resident = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                memory.append(resident)
                print(json.dumps({"soak_reads": i + 1, "resident_bytes": resident}), flush=True)
    if descriptors() != before:
        raise ValueError("File descriptors did not return to baseline")
    # Allow native allocator warm-up while bounding growth after the first 1k reads.
    if len(memory) > 1 and max(memory[1:]) - memory[0] > 96 * 1024**2:
        raise ValueError("Reader soak exceeded the predeclared 96 MiB growth budget")
    report.update(passed=True, sampled_content_sha256=digest.hexdigest(),
                  soak_reads=args.soak_reads, maximum_live_handles=high_handles,
                  resident_bytes=memory, descriptor_baseline=before,
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
