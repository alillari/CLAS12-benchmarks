#!/usr/bin/env python3
"""Measure trained noise models on GB10 using validation events only."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.graph_noise import Split, benchmark_forward, make_model, normalized, sync
from clas12_benchmarks.hdf5_cohort import write_json


def profile(root, study, output, device):
    torch.set_float32_matmul_precision("highest")
    receipt_path = root / "cohort.json"
    cohort_sha = file_sha256(receipt_path)
    report = json.loads((study / "summary.json").read_text())
    if cohort_sha != report["cohort_sha256"]:
        raise ValueError("Checkpoint study and profiling cohort differ")
    data = Split(root, json.loads(receipt_path.read_text()), "val")
    raw = data.features(None)
    results = {}
    for name, tagger in report["arms"].items():
        path = study / name / "checkpoint.pt"
        if file_sha256(path) != tagger["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed for {name}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        arm, batch_size = saved["arm"], saved["settings"]["batch_size"]
        model = make_model(arm, saved["scales"]).to(device)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        if unexpected or any(not key.startswith("backbone.") for key in missing):
            raise ValueError(f"Checkpoint parameters differ for {name}")
        sage = arm["model"]["architecture"] == "sage"
        features = normalized(raw, saved["scales"])

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        inference = benchmark_forward(model, data, features, batch_size, device, sage, warm_each_batch=True)
        inference_memory = int(torch.cuda.max_memory_allocated(device))
        inference_reserved = int(torch.cuda.max_memory_reserved(device))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=arm["lr"], weight_decay=1e-4)
        weights = tagger["class_weights"]
        if weights is not None:
            weights = torch.tensor(weights, device=device, dtype=torch.float32)
        model.train()
        rng = np.random.default_rng(20)
        total_points, total_seconds = 0, 0.
        indices = rng.integers(len(data.sizes), size=(23, batch_size))
        torch.cuda.reset_peak_memory_stats(device)
        for step, rows in enumerate(indices):
            sync(device)
            start = perf_counter()
            x, relation, y = data.batch(rows, features, device, sage)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, relation)
            loss = F.cross_entropy(logits.reshape(-1, 2), y.flatten(), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            sync(device)
            if step >= 3:
                total_points += int(data.sizes[rows].sum())
                total_seconds += perf_counter() - start
        results[name] = dict(inference=inference, peak_forward_gpu_bytes=inference_memory,
                             peak_forward_gpu_reserved_bytes=inference_reserved,
                             peak_training_gpu_bytes=int(torch.cuda.max_memory_allocated(device)),
                             peak_training_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                             training_clusters_per_s=total_points / total_seconds,
                             training_events_per_s=(len(indices) - 3) * batch_size / total_seconds,
                             timed_training_updates=len(indices) - 3, checkpoint_sha256=tagger["checkpoint_sha256"])
        print(f"Profiled {name}", flush=True)
        del optimizer, model, trainable
    write_json(output, dict(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                           timing_data="validation events, no validation or test metrics",
                           timing_scope=("CUDA synchronized. Each inference batch has an untimed warm-up. "
                                         "Training includes batch assembly and transfer."),
                           cohort_sha256=cohort_sha, arms=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profile(args.cohort, args.study, args.output, torch.device("cuda:0"))
