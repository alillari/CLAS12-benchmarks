"""Fit or apply a compact, standalone CVT cluster momentum MLP."""

import argparse
import csv
import gzip
import hashlib
import io
import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cluster_mlp_features import ARMS, INITIAL_COLUMNS, encode, rotate
from .cluster_momentum_geometry import GEOMETRY_SHAPE, MIN_LAYERS
from .data import file_sha256
from .momentum_manifest import TrackManifest

CHECKPOINTS = (1000, 3000, 6500, 10000, 15000, 22500, 30000, 37500, 45000)
SCHEMA = "clas12_cluster_mlp_v1"


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(320), nn.Linear(320, 640), nn.SiLU(),
                          nn.Dropout(.03), nn.Linear(640, 320))
            for _ in range(4)
        )
        self.norm = nn.LayerNorm(320)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x)


class ClusterMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.proj = nn.Linear(inputs, 320)
        self.layers = ResidualMLP()
        self.out = nn.Linear(320, 3)

    def forward(self, inputs):
        return self.out(self.layers(self.proj(inputs)))


def initial_model(arm, seed):
    torch.manual_seed(seed)
    # Draw in the original order so the retained coefficients and hidden layers
    # match the paired feature-selection study.
    model = ClusterMLP(307)
    columns = [INITIAL_COLUMNS.get(i, i) for i in ARMS[arm]]
    projection = nn.Linear(len(columns), 320)
    with torch.no_grad():
        projection.weight.copy_(model.proj.weight[:, columns])
        projection.bias.copy_(model.proj.bias)
    model.proj = projection
    return model


def target_coordinates(truth, frame):
    local = rotate(truth, frame)
    pt = local[:, :2].norm(dim=1)
    return torch.stack((local.norm(dim=1).log(), torch.atan2(local[:, 1], local[:, 0]),
                        torch.asinh(local[:, 2] / pt.clamp_min(1e-5))), 1)


def wrap(angle):
    return torch.atan2(angle.sin(), angle.cos())


def decode(standardized, frame, stats):
    logp, phi, eta = (standardized.float() * stats["y_std"] + stats["y_mean"]).unbind(1)
    p = logp.clamp(-6.5, 3).exp()
    eta = eta.clamp(-7, 7)
    pt = p / eta.cosh()
    return rotate(torch.stack((pt * phi.cos(), pt * phi.sin(), p * eta.tanh()), 1),
                  frame, inverse=True)


def geometry_tensors(batch, arm, device):
    raw = torch.tensor(batch, dtype=torch.float32, device=device)
    if raw.ndim != 3 or raw.shape[1:] not in (GEOMETRY_SHAPE, (12, 19)):
        raise ValueError("Expected layer geometry with shape [tracks,12,10] or archived [tracks,12,19]")
    if raw.shape[-1] == 19:
        raw = raw[..., [*range(9), 18]]
    count = raw[..., 9]
    if (not torch.isfinite(raw).all() or (count < 0).any()
            or ((count > 0).sum(1) < MIN_LAYERS).any()
            or (raw[..., :9][count == 0] != 0).any()):
        raise ValueError("Invalid layer geometry or fewer than four occupied layers")
    return encode(raw, arm)


def pack(path):
    tracks = TrackManifest(path)
    if not hasattr(tracks, "cluster_layer_geometry"):
        raise ValueError("Prepare an HDF5 momentum pack with --cluster-mlp-geometry")
    return tracks


def training_arrays(args, role, labels=None):
    if args.manifest:
        tracks = args.tracks
        rows = tracks.rows(role)[:labels]
        bounds = slice(int(rows[0]), int(rows[-1]) + 1) if len(rows) else slice(0, 0)
        source = {"manifest_sha256": tracks.sha256, "role": role, "tracks": len(rows)}
        return tracks.cluster_layer_geometry[bounds], tracks.truth[bounds], source
    split = "train" if role == "train" else "validation"
    x = args.study_cache / f"{split}_cluster.npy"
    y = args.study_cache / f"{split}_truth.npy"
    geometry, truth = (np.load(path, mmap_mode="r", allow_pickle=False) for path in (x, y))
    if len(geometry) != len(truth):
        raise ValueError("Study geometry and truth must have the same row count")
    source = {"geometry_sha256": file_sha256(x), "truth_sha256": file_sha256(y),
              "tracks": len(truth[:labels])}
    return geometry[:labels], truth[:labels], source


def encode_training(data, truth, arm, device):
    if truth.shape != (len(data), 3) or len(data) < 2:
        raise ValueError("Training and validation each require at least two aligned truth tracks")
    truth = torch.tensor(truth, dtype=torch.float32, device=device)
    if not torch.isfinite(truth).all() or (truth.norm(dim=1) <= 0).any():
        raise ValueError("Training and validation require finite, positive truth momentum")
    features, frames = zip(*(geometry_tensors(data[i:i + 32768], arm, device)
                             for i in range(0, len(data), 32768)), strict=True)
    return torch.cat(features), torch.cat(frames), truth


@torch.inference_mode()
def batch_prediction(model, x, frame, stats, amp):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        result = model(x)
    return decode(result, frame, stats)


@contextmanager
def new_output(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode=mode, dir=path.parent) as stream:
        yield stream
        stream.flush()
        os.link(stream.name, path)


def fit(args, device, amp):
    if args.output.exists():
        raise FileExistsError(args.output)
    scarce = args.arm == "bst_angle17"
    labels = (10000 if scarce else None) if args.labels is None else (
        None if args.labels == "all" else int(args.labels))
    steps = args.steps if args.steps is not None else (6500 if scarce else 45000)
    batch = args.batch if args.batch is not None else (256 if scarce else 512)
    if (labels is not None and labels < 2) or not 1 <= steps <= 45000 or batch < 1:
        raise ValueError("Require at least two labels, 1–45000 updates, and a positive batch size")
    if args.manifest:
        args.tracks = pack(args.manifest)
    geometry, truth, training_source = training_arrays(args, "train", labels)
    if labels is not None and len(truth) != labels:
        raise ValueError(f"Requested {labels} training labels but only {len(truth)} are available")
    x, frame, truth = encode_training(geometry, truth, args.arm, device)
    geometry, truth_val, validation_source = training_arrays(args, "validation")
    xv, fv, truth_val = encode_training(geometry, truth_val, args.arm, device)

    target = target_coordinates(truth, frame)
    p = truth.norm(dim=1)
    eta = torch.asinh(truth[:, 2] / truth[:, :2].norm(dim=1).clamp_min(1e-5))
    stats = {"x_mean": x.mean(0), "x_std": x.std(0).clamp_min(.025),
             "y_mean": torch.stack((p.log().mean(), p.new_zeros(()), eta.mean())),
             "y_std": torch.stack((p.log().std(), p.new_tensor(.22), eta.std())).clamp_min(.05)}
    x.sub_(stats["x_mean"]).div_(stats["x_std"])
    xv.sub_(stats["x_mean"]).div_(stats["x_std"])
    target = (target - stats["y_mean"]) / stats["y_std"]
    weight = p.clamp(.3, 7).sqrt()
    weight /= weight.mean()
    weight = torch.stack((1.7 * weight, torch.full_like(weight, .65),
                          torch.full_like(weight, .75)), 1)

    model = initial_model(args.arm, args.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00065, weight_decay=.003,
                                  fused=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed + 397)
    draws = torch.randint(len(truth), (steps, batch), generator=generator, device=device)
    torch.manual_seed(args.seed + 910)
    checkpoints = {point for point in CHECKPOINTS if point <= steps} | {steps}
    p_val = truth_val.norm(dim=1)
    best, checkpoint, history = math.inf, None, []
    for step, indices in enumerate(draws, 1):
        model.train()
        optimizer.param_groups[0]["lr"] = .00065 * min(
            step / 350, .08 + .92 * .5 * (1 + math.cos(math.pi * step / 45000)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            prediction = model(x[indices])
        dp, dphi, deta = (prediction.float() - target[indices]).unbind(1)
        scale = stats["y_std"][1]
        errors = torch.stack((dp, wrap(dphi * scale) / scale, deta), 1)
        loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=.05, reduction="none")
        (loss * weight[indices]).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=True)
        optimizer.step()
        if step in checkpoints:
            model.eval()
            vectors = torch.cat([batch_prediction(model, xb, fb, stats, amp)
                                 for xb, fb in zip(xv.split(1024), fv.split(1024), strict=True)])
            score = float((vectors.norm(dim=1) - p_val).abs().mean())
            history.append({"step": step, "p_mae_gev": score})
            print(f"Update {step}  validation p MAE {score * 1000:.3f} MeV/c", flush=True)
            if score < best:
                best = score
                checkpoint = {"step": step, "state": {k: v.detach().cpu().clone()
                                                      for k, v in model.state_dict().items()}}
    if checkpoint is None:
        raise ValueError("No finite validation checkpoint")
    checkpoint.update(schema=SCHEMA, arm=args.arm, columns=ARMS[args.arm],
                      params={k: v.cpu() for k, v in stats.items()}, seed=args.seed,
                      labels=len(truth), steps=steps, batch=batch, history=history,
                      draw_sha256=hashlib.sha256(draws.cpu().numpy().tobytes()).hexdigest(),
                      training_source=training_source, validation_source=validation_source,
                      torch=str(torch.__version__), device=str(device), bf16=amp)
    with new_output(args.output, "wb") as stream:
        torch.save(checkpoint, stream)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    arm = checkpoint["arm"]
    if (checkpoint.get("schema") not in (None, SCHEMA) or arm not in ARMS
            or checkpoint["columns"] != ARMS[arm]):
        raise ValueError("Checkpoint recipe or feature order is incompatible with the cluster MLP")
    stats = checkpoint["params"]
    for key, size in {"x_mean": len(ARMS[arm]), "x_std": len(ARMS[arm]), "y_mean": 3, "y_std": 3}.items():
        if (stats[key].shape != (size,) or not torch.isfinite(stats[key]).all()
                or (key.endswith("_std") and (stats[key] <= 0).any())):
            raise ValueError(f"Invalid checkpoint normalization: {key}")
    model = ClusterMLP(len(ARMS[arm])).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    return arm, model, stats


def prediction_batches(geometry, checkpoint, device, amp):
    arm, model, stats = load_checkpoint(checkpoint, device)
    for start in range(0, len(geometry), 1024):
        x, frame = geometry_tensors(geometry[start:start + 1024], arm, device)
        x = (x - stats["x_mean"]) / stats["x_std"]
        values = batch_prediction(model, x, frame, stats, amp).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Checkpoint produced nonfinite predictions")
        yield values


def predict_pack(args, device, amp):
    tracks = pack(args.manifest)
    rows = tracks.rows(args.role)
    if not len(rows):
        raise ValueError(f"No tracks in role {args.role!r}")
    fields = (*tracks.identity_fields, "schema", "momentum_unit", "manifest_sha256",
              "pred_px_gev", "pred_py_gev", "pred_pz_gev")
    contract = ("clas12_momentum_predictions_v2", "GeV", tracks.sha256)
    with new_output(args.output, "wb") as stream:
        binary = gzip.GzipFile(fileobj=stream, mode="wb") if args.output.suffix == ".gz" else stream
        output = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        try:
            writer = csv.writer(output)
            writer.writerow(fields)
            start = 0
            geometry = tracks.cluster_layer_geometry[rows[0]:rows[-1] + 1]
            for values in prediction_batches(geometry, args.checkpoint, device, amp):
                for i, p in zip(rows[start:start + len(values)], values, strict=True):
                    writer.writerow((*tracks.identity(i), *contract, *p))
                start += len(values)
        finally:
            output.flush()
            output.detach()
            if binary is not stream:
                binary.close()


def predict_geometry(args, device, amp):
    geometry = np.load(args.geometry, mmap_mode="r", allow_pickle=False)
    if not len(geometry):
        raise ValueError("Geometry file has no tracks")
    results = np.concatenate(list(prediction_batches(geometry, args.checkpoint, device, amp)))
    with new_output(args.output, "wb") as stream:
        np.save(stream, results, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Select checkpoint by validation momentum MAE")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--study-cache", type=Path)
    train.add_argument("--arm", choices=ARMS, default="g18_endz")
    train.add_argument("--labels", help="Number of training rows from the frozen prefix, or all")
    train.add_argument("--steps", type=int)
    train.add_argument("--batch", type=int)
    train.add_argument("--seed", type=int, default=23)
    predict = sub.add_parser("predict", help="Write keyed HDF5 benchmark predictions")
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--role", default="evaluation")
    geometry = sub.add_parser("predict-geometry", help="Apply a checkpoint to an NPY array of layer means")
    geometry.add_argument("--geometry", type=Path, required=True)
    for command in (predict, geometry):
        command.add_argument("--checkpoint", type=Path, required=True)
    for command in (train, predict, geometry):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--fp32", action="store_true", help="Disable CUDA BF16 autocast")
    args = parser.parse_args()
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.fp32 and torch.cuda.is_bf16_supported()
    {"train": fit, "predict": predict_pack, "predict-geometry": predict_geometry}[args.command](args, device, amp)


if __name__ == "__main__":
    main()
