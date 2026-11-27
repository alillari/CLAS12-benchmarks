"""Cache frozen M4 final-state features using observed coordinates only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import file_sha256
from .graph_noise import Split
from .graph_noise_fm import FMNoiseClassifier
from .hdf5_cohort import publication, write_json


def freeze(cohort, output, checkpoint, sha256, train_events, batch_size):
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    receipt = json.loads((cohort / "cohort.json").read_text())
    model = FMNoiseClassifier({"eta": 1., "radius": 1.}, checkpoint, sha256).to(device).eval()
    counts = dict(train=train_events, val=receipt["splits"]["val"]["events"],
                  test=receipt["splits"]["test"]["events"])
    results = {}
    with publication(output, manifest_name="features.json") as stage, torch.inference_mode():
        for role, count in counts.items():
            split = Split(cohort, receipt, role)
            if not 1 <= count <= len(split.sizes):
                raise ValueError(f"Invalid requested event count for {role}")
            coords = split.features(count)
            path = stage / f"{role}.npy"
            buffer = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                               shape=(split.offsets[count], 512))
            order = np.argsort(split.sizes[:count], kind="stable")
            for start in range(0, count, batch_size):
                ids = order[start:start + batch_size]
                if not split.sizes[ids].sum():
                    continue
                x, mask, _ = split.batch(ids, coords, device, sage=False)
                values = model.extract_features(x, mask).cpu().numpy()
                for row, index in enumerate(ids):
                    begin, end = split.offsets[index:index + 2]
                    buffer[begin:end] = values[row, :end - begin]
                if start == 0:
                    cached = FMNoiseClassifier({"eta": 1., "radius": 1.}, cached=True).to(device).eval()
                    cached.head.load_state_dict(model.head.state_dict())
                    torch.testing.assert_close(cached(torch.as_tensor(values, device=device), mask),
                                               model(x, mask), atol=.0002, rtol=.001)
            for start in range(0, len(buffer), 65536):
                if not np.isfinite(buffer[start:start + 65536]).all():
                    raise RuntimeError(f"Nonfinite frozen features in {role}")
            buffer.flush()
            del buffer
            results[role] = {"events": count, "sha256": file_sha256(path)}
            print(f"Cached frozen M4 for {count} {role} events", flush=True)
        write_json(stage / "features.json", dict(cohort_sha256=file_sha256(cohort / "cohort.json"),
                                                checkpoint_sha256=sha256, dtype="float32", splits=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--train-events", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    freeze(**vars(args))
