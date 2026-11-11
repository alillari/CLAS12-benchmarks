"""Unmodified, pinned data-only helpers from the committed CLAS12 FM work."""
import json
from pathlib import Path

from ..data import file_sha256


def source_receipt(*filenames):
    directory = Path(__file__).parent
    receipt = json.loads((directory / "sources.json").read_text())
    receipt["files"] = {name: receipt["files"][name] for name in filenames}
    for name, item in receipt["files"].items():
        if file_sha256(directory / name) != item["sha256"]:
            raise ValueError(f"Pinned FM reference changed: {name}")
    return receipt
