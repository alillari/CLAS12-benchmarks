#!/usr/bin/env python3
"""Compare every imported file byte-for-byte with its pinned upstream Git blob."""
import hashlib
import json
from pathlib import Path
import subprocess


def verify(root):
    lock = json.loads((root / "sources.lock.json").read_text())
    results = []
    for source in lock["sources"]:
        entries = subprocess.check_output(
            ["git", "ls-tree", "-rz", source["commit"]], cwd=root).split(b"\0")
        expected = set()
        for entry in entries:
            if not entry:
                continue
            header, encoded_name = entry.split(b"\t", 1)
            mode, kind, blob = header.decode().split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError(f"Unsupported imported entry: {entry!r}")
            name = encoded_name.decode()
            path = root / source["directory"] / name
            content = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if actual != blob:
                raise ValueError(f"Source differs from pinned upstream: {path}")
            if (path.stat().st_mode & 0o111 != 0) != (mode == "100755"):
                raise ValueError(f"Source executable bit differs: {path}")
            expected.add(name)
        actual_files = {str(path.relative_to(root / source["directory"]))
                        for path in (root / source["directory"]).rglob("*") if path.is_file()}
        if actual_files != expected:
            raise ValueError(f"Unexpected/missing files in {source['directory']}: {actual_files ^ expected}")
        results.append({"source": source["name"], "commit": source["commit"], "verified_files": len(expected)})
    return results


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
