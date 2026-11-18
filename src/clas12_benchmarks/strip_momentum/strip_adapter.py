"""AdapterOnly momentum fits from measured strip geometry, with no backbone."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cache_strips import FEATURES, LAYERS

CENTER = "center"
GEOMETRY = "geometry"
STEREO = "stereo"
HYBRID = "hybrid"
HELIX = "helix"
KINDS = (CENTER, GEOMETRY, STEREO, HYBRID, HELIX)


def write_json(path, payload):
    swap = path.with_suffix(path.suffix + ".tmp")
    swap.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    swap.replace(path)


def rotate_xyz(x, frame, inverse=False):
    c, s = frame.unbind(-1)
    for _ in range(x.ndim - frame.ndim):
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    if inverse:
        s = -s
    a, b, z = x.unbind(-1)
    return torch.stack((c * a + s * b, c * b - s * a, z), -1)


def stereo_points(layer, present):
    """Closest points on charge-weighted strips from the two BST sensor faces."""
    a, c = layer[:, 0:6:2, 3:6], layer[:, 1:6:2, 3:6]
    b, d = layer[:, 0:6:2, 6:9], layer[:, 1:6:2, 6:9]
    u, v, w = b - a, d - c, a - c
    dot = lambda a, b: (a * b).sum(-1)
    uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
    uw, vw = dot(u, w), dot(v, w)
    determinant = uu * vv - uv.square()
    den = determinant.clamp_min(1e-9)
    t = (uv * vw - vv * uw) / den
    s = (uu * vw - uv * uw) / den
    x, y = a + t[..., None] * u, c + s[..., None] * v
    distance = torch.linalg.vector_norm(x - y, dim=-1)
    valid = present[:, 0:6:2] & present[:, 1:6:2] & (determinant > 1e-6)
    valid &= (t > -.2) & (t < 1.2) & (s > -.2) & (s < 1.2) & (distance < .2)
    return (x + y) * .5, distance, valid


def line_summary(x, y, mask):
    weights = mask.float()
    n = weights.sum(1, keepdim=True)
    mu_x = (x * weights).sum(1, keepdim=True) / n.clamp_min(1)
    mu_y = (y * weights).sum(1, keepdim=True) / n.clamp_min(1)
    variance = ((x - mu_x).square() * weights).sum(1, keepdim=True)
    slope = (((x - mu_x) * (y - mu_y)) * weights).sum(1, keepdim=True) / variance.clamp_min(1e-5)
    slope = slope.clamp(-4, 4)
    scatter = ((((x - mu_x) * slope + mu_y - y).square() * weights)
               .sum(1, keepdim=True) / n.clamp_min(1)).sqrt()
    return torch.cat((n, mu_x, mu_y, slope, scatter), 1)


def bending_features(points, mask, z_points, z_mask):
    """Estimate signed bending using the beam axis and pairs of measured XY points."""
    first, second = torch.triu_indices(points.size(1), points.size(1), 1, device=points.device)
    a, b = points[:, first], points[:, second]
    r = torch.linalg.vector_norm(points, dim=-1)
    ra, rb = r[:, first], r[:, second]
    chord = torch.linalg.vector_norm(b - a, dim=-1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    signed_separation = torch.sign(rb - ra)
    valid = mask[:, first] & mask[:, second] & ((rb - ra).abs() > .09)
    curvature = (2 * cross * signed_separation / (ra * rb * chord).clamp_min(1e-6)).clamp(-2, 2)
    curvature = curvature.masked_fill(~valid, 0)
    weight = ((rb - ra).square() * valid)
    k = (curvature * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1e-6)
    spread = (((curvature - k).square() * weight).sum(1, keepdim=True)
              / weight.sum(1, keepdim=True).clamp_min(1e-6)).sqrt()
    alpha = (r * k * .5).clamp(-.99, .99).asin()
    phi0 = torch.atan2(points[..., 1], points[..., 0]) - alpha
    phase = torch.cat((phi0.sin() * mask, phi0.cos() * mask), 1)
    z_radius = torch.linalg.vector_norm(z_points[..., :2], dim=-1)
    arc = (z_radius * k.abs() * .5).clamp_max(.99).asin() * 2 / k.abs().clamp_min(1e-6)
    arc = torch.where(k.abs() > 1e-6, arc, z_radius)
    longitudinal = line_summary(arc, z_points[..., 2], z_mask)
    return torch.cat((curvature, valid, phase, k, spread, longitudinal), 1)


def features(layer):
    present = layer[..., 18] > 0
    bmt_z, bmt_c = [7, 8, 10], [6, 9, 11]
    transverse = present.clone()
    transverse[:, bmt_c] = False
    trans_weights = transverse | ~transverse.any(1, keepdim=True)
    anchor = (layer[..., 15:17] * trans_weights[..., None]).sum(1)
    angle = torch.atan2(anchor[:, 1], anchor[:, 0])
    frame = torch.stack((angle.cos(), angle.sin()), 1)

    center = rotate_xyz(layer[..., 15:18], frame)
    center[..., 1] *= 8
    center = torch.cat((center, present[..., None]), -1).flatten(1)
    local = torch.cat([rotate_xyz(layer[..., i:i + 3], frame) for i in (0, 3, 6)]
                      + [layer[..., 9:]], -1)
    for i in (1, 4, 7):
        local[..., i] *= 8
    geometry = local.flatten(1)

    xy_z, distance, stereo_mask = stereo_points(layer, present)
    xy_z = rotate_xyz(xy_z, frame)
    stereo = torch.cat((xy_z * xy_z.new_tensor([1, 8, 1]), distance[..., None]),
                       -1).masked_fill(~stereo_mask[..., None], 0)
    xyz = rotate_xyz(layer[..., :3], frame)
    r = torch.linalg.vector_norm(xyz[..., :2], dim=-1)
    rho = torch.linalg.vector_norm(xy_z[..., :2], dim=-1)
    phi = torch.atan2(xyz[..., 1], xyz[..., 0])
    psi = torch.atan2(xy_z[..., 1], xy_z[..., 0])
    xy_fit = line_summary(torch.cat((rho, r[:, bmt_z]), 1),
                          torch.cat((psi, phi[:, bmt_z]), 1),
                          torch.cat((stereo_mask, present[:, bmt_z]), 1))
    wide_fit = line_summary(r, phi, transverse)
    z_fit = line_summary(torch.cat((rho, r[:, bmt_c]), 1),
                         torch.cat((xy_z[..., 2], xyz[:, bmt_c, 2]), 1),
                         torch.cat((stereo_mask, present[:, bmt_c]), 1))
    extra = torch.cat((stereo.flatten(1), stereo_mask.float(), xy_fit, wide_fit, z_fit), 1)
    stereo_features = torch.cat((geometry, extra), 1)
    bending = bending_features(
        torch.cat((xy_z[..., :2], xyz[:, bmt_z, :2]), 1),
        torch.cat((stereo_mask, present[:, bmt_z]), 1),
        torch.cat((xy_z, xyz[:, bmt_c]), 1),
        torch.cat((stereo_mask, present[:, bmt_c]), 1),
    )
    return frame, {
        CENTER: center,
        GEOMETRY: geometry,
        STEREO: stereo_features,
        HELIX: torch.cat((stereo_features, bending), 1),
    }


class ResidualMLP(nn.Module):
    def __init__(self, width, depth, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                          nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 2, width))
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x)


class StripAdapter(nn.Module):
    def __init__(self, kind, inputs, width, depth, dropout):
        super().__init__()
        self.kind = kind
        if kind not in KINDS:
            raise ValueError(f"Unsupported strip model {kind!r}")
        self.proj = nn.Linear(inputs, width)
        if kind == HYBRID:
            self.raw_proj = nn.Linear(FEATURES, 64)
            self.layer_embedding = nn.Embedding(LAYERS + 1, 64)
            self.encoder = nn.TransformerEncoderLayer(
                64, 4, 128, dropout, batch_first=True, activation="gelu", norm_first=True)
            self.raw_fusion = nn.Linear(64, width)
        self.layers = ResidualMLP(width, depth, dropout)
        self.out = nn.Linear(width, 3)

    def forward(self, x, frame, raw=None, layer=None):
        x = self.proj(x)
        if self.kind == HYBRID:
            mask = layer >= 0
            raw = raw.float()
            local = torch.cat([rotate_xyz(raw[..., j:j + 3], frame) for j in (0, 3, 6)]
                              + [raw[..., 9:]], -1)
            emb = self.raw_proj(local) + self.layer_embedding(layer.long() + 1)
            emb = self.encoder(emb, src_key_padding_mask=~mask)
            pooled = (emb * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            x = x + self.raw_fusion(pooled)
        return self.out(self.layers(x))


def load_data(root, split, limit, need_raw, device, feature_kinds=None):
    def read(field):
        return np.load(root / f"{split}_{field}.npy", mmap_mode="c")[:limit]

    report = json.loads((root / f"{split}.json").read_text())
    y = torch.from_numpy(read("truth")).to(device)
    layer_array = read("layer")
    if feature_kinds is not None:
        feature_kinds = {STEREO if kind == HYBRID else kind for kind in feature_kinds}
    frame_parts, feature_parts = [], []
    for start in range(0, len(y), 32768):
        layer = torch.from_numpy(layer_array[start:start + 32768]).to(device)
        frame, flat = features(layer)
        frame_parts.append(frame)
        feature_parts.append(flat if feature_kinds is None else
                             {key: flat[key] for key in flat.keys() & feature_kinds})
    data = {
        "truth": y, "frame": torch.cat(frame_parts), "report": report,
        "features": {k: torch.cat([p[k] for p in feature_parts]) for k in feature_parts[0]},
        "identity": read("identity"), "pid": read("pid"), "coverage": read("coverage"),
        "count": read("count"), "coatjava": read("coatjava"),
    }
    if STEREO in data["features"]:
        data["features"][HYBRID] = data["features"][STEREO]
    if need_raw:
        data.update(raw=torch.from_numpy(read("raw")).to(device),
                    layer=torch.from_numpy(read("raw_layer")).to(device))
    return data


def standardized_params(train, kind, loss):
    frame, cart = train["frame"], train["truth"]
    x = train["features"][kind]
    scale = x.std(0).clamp_min(.025)
    mean = x.mean(0)
    if loss.startswith("angular"):
        p = torch.linalg.vector_norm(cart, dim=1)
        eta = torch.asinh(cart[:, 2] / torch.linalg.vector_norm(cart[:, :2], dim=1).clamp_min(1e-5))
        center, std = p.log().mean(), p.log().std()
        y_mean = torch.stack((center, p.new_zeros(()), eta.mean()))
        y_std = torch.stack((std, p.new_tensor(.22), eta.std())).clamp_min(.05)
    else:
        values = rotate_xyz(cart, frame)
        y_mean, y_std = values.mean(0), values.std(0).clamp_min(.04)
    return dict(x_mean=mean, x_std=scale, y_mean=y_mean, y_std=y_std)


def targets(truth, frame, loss, params):
    if loss.startswith("angular"):
        local = rotate_xyz(truth, frame)
        p = torch.linalg.vector_norm(local, dim=1)
        pt = torch.linalg.vector_norm(local[:, :2], dim=1)
        coordinates = torch.stack((p.log(), torch.atan2(local[:, 1], local[:, 0]),
                                   torch.asinh(local[:, 2] / pt.clamp_min(1e-5))), 1)
    else:
        coordinates = rotate_xyz(truth, frame)
    return (coordinates - params["y_mean"]) / params["y_std"]


def physical(pred, frame, loss, params):
    values = pred.float() * params["y_std"] + params["y_mean"]
    if loss.startswith("angular"):
        logp, phi, eta = values.unbind(1)
        p = logp.clamp(-6.5, 3).exp()
        eta = eta.clamp(-7, 7)
        pt = p / eta.cosh()
        values = torch.stack((pt * phi.cos(), pt * phi.sin(), p * eta.tanh()), 1)
    return rotate_xyz(values, frame, inverse=True)


def infer(model, data, index, params, amp):
    kind = model.kind
    x = (data["features"][kind][index] - params["x_mean"]) / params["x_std"]
    raw = layer = None
    if kind == HYBRID:
        raw, layer = data["raw"][index], data["layer"][index]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        return model(x, data["frame"][index], raw, layer).float()


def errors(truth, pred):
    p, pp = (np.linalg.norm(x, axis=1) for x in (truth, pred))
    dp = pp - p
    ratio = dp / p
    theta, theta_pred = (np.arctan2(np.linalg.norm(x[:, :2], axis=1), x[:, 2])
                         for x in (truth, pred))
    phi, phi_pred = (np.arctan2(x[:, 1], x[:, 0]) for x in (truth, pred))
    dphi = (phi_pred - phi + np.pi) % (2 * np.pi) - np.pi
    return dp, ratio, theta_pred - theta, dphi


def metrics(truth, pred):
    if not len(truth):
        return {"tracks": 0}
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions")
    dp, ratio, dtheta, dphi = errors(truth, pred)
    q16, _, q84 = np.quantile(ratio, [.16, .5, .84])
    return dict(
        tracks=len(truth), p_mae_gev=float(np.abs(dp).mean()), p_rmse_gev=float(np.mean(dp ** 2)**.5),
        relative_bias=float(ratio.mean()), relative_median=float(np.median(ratio)),
        relative_c68=float((q84 - q16) / 2), relative_abs_median=float(np.median(np.abs(ratio))),
        relative_abs_mean=float(np.mean(np.abs(ratio))), relative_tail10=float((np.abs(ratio) > .1).mean()),
        theta_mae_deg=float(np.degrees(np.abs(dtheta).mean())),
        phi_mae_deg=float(np.degrees(np.abs(dphi).mean())),
        cartesian_mae_gev=float(np.abs(truth - pred).mean()),
    )


@torch.inference_mode()
def predict(model, data, loss, params, amp):
    model.eval()
    predictions = []
    for index in torch.arange(len(data["truth"]), device=data["truth"].device).split(512):
        pred = infer(model, data, index, params, amp)
        predictions.append(physical(pred, data["frame"][index], loss, params))
    return torch.cat(predictions).cpu().numpy()


def train_fit(args, arm, train, validation):
    kind, loss, rate, width, depth, seed = arm
    name = f"{kind}-{loss}-lr{rate:g}-w{width}-d{depth}-s{seed}"
    folder = args.out / name
    folder.mkdir()
    torch.manual_seed(seed)
    device = train["truth"].device
    params = standardized_params(train, kind, loss)
    model_args = (kind, len(params["x_mean"]), width, depth, args.dropout)
    model = StripAdapter(*model_args).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=args.decay, fused=True)
    generator = torch.Generator(device=device).manual_seed(seed + 397)
    standardized = targets(train["truth"], train["frame"], loss, params)
    mag_weight = torch.linalg.vector_norm(train["truth"], dim=1).clamp(.3, 7).sqrt()
    mag_weight /= mag_weight.mean()
    truth = validation["truth"].cpu().numpy()
    history = []
    best = float("inf")
    start = time.monotonic()
    print(f"START {name} params={count} tracks={len(standardized)} steps={args.steps}", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        fraction = step / max(1, min(350, args.steps // 10))
        cosine = .08 + .92 * .5 * (1 + math.cos(math.pi * step / args.steps))
        optimizer.param_groups[0]["lr"] = rate * min(fraction, cosine)
        idx = torch.randint(len(standardized), (args.batch,), device=device, generator=generator)
        prediction = infer(model, train, idx, params, args.amp)
        delta = prediction - standardized[idx]
        if loss.startswith("angular"):
            phi_scale = params["y_std"][1]
            dphi = delta[:, 1] * phi_scale
            wrapped = torch.atan2(dphi.sin(), dphi.cos()) / phi_scale
            delta = torch.stack((delta[:, 0], wrapped, delta[:, 2]), 1)
            objective = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=.05, reduction="none")
            objective = objective * delta.new_tensor([1.7, .65, .75])
            if loss == "angular-weighted":
                objective = torch.cat((objective[:, :1] * mag_weight[idx, None], objective[:, 1:]), 1)
            value = objective.mean()
        else:
            value = delta.abs().mean()
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % args.interval and step != args.steps:
            continue
        pred = predict(model, validation, loss, params, args.amp)
        score = metrics(truth, pred)
        row = dict(step=step, seconds=round(time.monotonic() - start, 1),
                   minibatch_loss=float(value.detach()), **score)
        history.append(row)
        print(json.dumps({"arm": name, **row}), flush=True)
        if score["p_mae_gev"] < best:
            best = score["p_mae_gev"]
            checkpoint = dict(model_args=model_args, loss=loss, amp=args.amp, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                              state=model.state_dict(), stats=params, step=step, validation=score,
                              manifest_sha256=train["report"]["manifest_sha256"])
            torch.save(checkpoint, folder / "best.pth")
            np.save(folder / "validation_predictions.npy", pred)
        write_json(folder / "history.json", history)
    best_row = min(history, key=lambda x: x["p_mae_gev"])
    result = dict(arm=name, model=kind, loss=loss, lr=rate, width=width, depth=depth,
                  seed=seed, parameters=count, best=best_row)
    write_json(folder / "result.json", result)
    return result


def parse_arm(text):
    kind, loss, rate, width, depth, seed = text.split(":")
    if kind not in KINDS or loss not in ("angular", "angular-weighted", "cartesian"):
        raise argparse.ArgumentTypeError(f"Unsupported strip model or loss: {text}")
    return kind, loss, float(rate), int(width), int(depth), int(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/cache"))
    parser.add_argument("--val-cache", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/results"))
    parser.add_argument("--arm", type=parse_arm, action="append", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--val-size", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=.03)
    parser.add_argument("--decay", type=float, default=.003)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    raw = any(arm[0] == HYBRID for arm in args.arm)
    kinds = {arm[0] for arm in args.arm}
    train = load_data(args.cache, "train", args.train_size, raw, device, kinds)
    val = load_data(args.val_cache or args.cache, "validation", args.val_size, raw, device, kinds)
    assert len(train["truth"]) == args.train_size and len(val["truth"]) == args.val_size
    assert set(train["identity"][:, 0]).isdisjoint(val["identity"][:, 0])
    assert train["report"]["manifest_sha256"] == val["report"]["manifest_sha256"]
    manifest = dict(train=train["report"], validation=val["report"],
                    train_tracks=args.train_size, validation_tracks=args.val_size,
                    train_prefix_sha256=hashlib.sha256(train["identity"].tobytes()).hexdigest(),
                    val_prefix_sha256=hashlib.sha256(val["identity"].tobytes()).hexdigest(),
                    target="observed direct primary (px,py,pz), GeV/c",
                    selection="lowest validation mean absolute error in p (GeV/c)")
    write_json(args.out / "cohort.json", manifest)
    summary = []
    for arm in args.arm:
        summary.append(train_fit(args, arm, train, val))
        summary.sort(key=lambda x: x["best"]["p_mae_gev"])
        write_json(args.out / "ranking.json", summary)


if __name__ == "__main__":
    main()
