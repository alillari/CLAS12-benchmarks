"""Shared source and runtime receipts for the noise benchmarks."""

from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

import torch

from .data import file_sha256


def run_provenance(device):
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    files = sorted(package.rglob("*.py")) + [package / "fm_reference" / "sources.json"]
    source_lock = root / "sources.lock.json"
    packages = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages.setdefault(re.sub(r"[-_.]+", "-", name).lower(), dist.version)
    receipt = {
        "source_sha256": {p.relative_to(package).as_posix(): file_sha256(p) for p in files},
        "sources_lock": json.loads(source_lock.read_text()) if source_lock.is_file() else None,
        "git_commit": os.environ.get("BENCHMARK_GIT_COMMIT"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if (root / ".git").exists() and shutil.which("git"):
        git = ["git", "--no-optional-locks", "-C", str(root)]
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        patch = subprocess.check_output(
            [*git, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD",
             "--", "src", "scripts", "configs", "pyproject.toml", "sources.lock.json"],
            encoding="utf-8", errors="surrogateescape")
        receipt.update(git_commit=head, tracked_source_diff=patch)
    return receipt
