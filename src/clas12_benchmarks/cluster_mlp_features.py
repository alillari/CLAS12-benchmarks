"""Track-aligned inputs for the three compact cluster momentum recipes."""

import torch

BMT_Z = [7, 8, 10]
BMT_C = [6, 9, 11]

# Column order in the archived study encoder and checkpoint files.
BST_END_Y = [19 * layer + endpoint for layer in range(6) for endpoint in (4, 7)]
ARMS = {
    "g18_endz": BST_END_Y + [19 * layer + 1 for layer in BMT_Z]
    + [19 * layer + 2 for layer in BMT_C] + [19 * layer + 8 for layer in range(6)],
    "bmt_cslope_outer16": BST_END_Y + [320, 322, 323, 19 * 5 + 8],
    "bst_angle17": BST_END_Y + [326, 324, 325, 322, 323],
}
# The study inherited these initial coefficients for derived inputs.
INITIAL_COLUMNS = {320: 19 * 8 + 1, 322: 19 * 9 + 2, 323: 256,
                   324: 305, 325: 245, 326: 300}


def rotate(xyz, frame, inverse=False):
    cos, sin = frame.unbind(-1)
    for _ in range(xyz.ndim - frame.ndim):
        cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)
    if inverse:
        sin = -sin
    x, y, z = xyz.unbind(-1)
    return torch.stack((cos * x + sin * y, cos * y - sin * x, z), -1)


def bmt_line(radius, value, present):
    count = present.sum(1).clamp_min(1)
    mean_r = (radius * present).sum(1) / count
    mean_y = (value * present).sum(1) / count
    dr = radius - mean_r[:, None]
    slope = ((dr * (value - mean_y[:, None]) * present).sum(1)
             / (dr.square() * present).sum(1).clamp_min(1e-6))
    slope = torch.where(count >= 2, slope, 0.)
    return mean_y + (.6 - mean_r) * slope, slope


def stereo_points(layer, present, frame):
    """Closest approach of the two mean cluster endpoint lines in each BST pair."""
    a, c = layer[:, :6:2, 3:6], layer[:, 1:6:2, 3:6]
    b, d = layer[:, :6:2, 6:9], layer[:, 1:6:2, 6:9]
    u, v, w = b - a, d - c, a - c
    uu, uv, vv = u.square().sum(-1), (u * v).sum(-1), v.square().sum(-1)
    uw, vw = (u * w).sum(-1), (v * w).sum(-1)
    determinant = uu * vv - uv.square()
    denom = determinant.clamp_min(1e-9)
    t = (uv * vw - vv * uw) / denom
    s = (uu * vw - uv * uw) / denom
    x, y = a + t[..., None] * u, c + s[..., None] * v
    distance = torch.linalg.vector_norm(x - y, dim=-1)
    valid = present[:, :6:2] & present[:, 1:6:2] & (determinant > 1e-6)
    valid &= (t > -.2) & (t < 1.2) & (s > -.2) & (s < 1.2) & (distance < .2)
    midpoint = rotate((x + y) * .5, frame)
    return midpoint.masked_fill(~valid[..., None], 0.), valid


def chord_inputs(layer, centers, present, frame):
    stereo, stereo_valid = stereo_points(layer, present, frame)
    xy = torch.cat((stereo[..., :2], centers[:, BMT_Z, :2]), 1)
    valid = torch.cat((stereo_valid, present[:, BMT_Z]), 1)
    radius = torch.linalg.vector_norm(xy, dim=-1)
    low = radius.masked_fill(~valid, torch.inf).argmin(1)
    high = radius.masked_fill(~valid, -torch.inf).argmax(1)
    rows = torch.arange(len(xy), device=xy.device)
    a, b = xy[rows, low], xy[rows, high]
    ra, rb = radius[rows, low], radius[rows, high]
    denom = (ra * rb * (b - a).norm(dim=1)).clamp_min(1e-6)
    curvature = (2 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / denom).clamp(-2, 2)
    curvature = torch.where((valid.sum(1) >= 2) & (rb - ra > .09), curvature, 0.)

    longitudinal = torch.cat((stereo, centers[:, BMT_C]), 1)
    z = longitudinal[..., 2]
    rz = torch.linalg.vector_norm(longitudinal[..., :2], dim=-1)
    k = curvature.abs()[:, None]
    arc = 2 * (rz * k * .5).clamp_max(.99).asin() / k.clamp_min(1e-6)
    arc = torch.where(k > 1e-6, arc, rz)
    base_weight = torch.cat((stereo_valid, present[:, BMT_C]), 1) * z.new_tensor([1, 1, 1, 4, 4, 4])
    weight = base_weight
    for _ in range(3):
        total = weight.sum(1).clamp_min(1)
        mean_arc = (weight * arc).sum(1) / total
        mean_z = (weight * z).sum(1) / total
        dr = arc - mean_arc[:, None]
        slope = ((weight * dr * (z - mean_z[:, None])).sum(1)
                 / (weight * dr.square()).sum(1).clamp_min(1e-6)).clamp(-4, 4)
        intercept = mean_z - slope * mean_arc
        residual = z - (slope[:, None] * arc + intercept[:, None])
        weight = base_weight * (.02 / residual.abs().clamp_min(.02))

    eta = torch.asinh(slope)
    bend = (radius * curvature[:, None] * .5).clamp(-.99, .99).asin()
    origin = torch.atan2(xy[..., 1], xy[..., 0]) - bend
    phi = torch.atan2((origin.sin() * valid).sum(1), (origin.cos() * valid).sum(1))
    first = layer[..., :2].norm(dim=-1).masked_fill(~present, torch.inf).amin(1)
    phi = phi + 2 * (first * curvature * .5).clamp(-.99, .99).asin()
    phi = torch.atan2(phi.sin(), phi.cos())
    logp = (eta.cosh() / curvature.abs().clamp_min(.03)).log()
    logp = torch.where((curvature != 0) & valid.any(1), logp, 0.)
    return torch.stack((logp, eta, phi), 1)


def encode(layer, arm):
    """Encode [N,12,10] measured layer geometry and return (inputs, frame)."""
    if arm not in ARMS:
        raise ValueError(f"Unsupported cluster MLP recipe: {arm}")
    present = layer[..., 9] > 0
    transverse = present.clone()
    transverse[:, BMT_C] = False
    anchor = (layer[..., :2] * (transverse | ~transverse.any(1, keepdim=True))[..., None]).sum(1)
    alpha = torch.atan2(anchor[:, 1], anchor[:, 0])
    frame = torch.stack((alpha.cos(), alpha.sin()), 1)
    centers = rotate(layer[..., :3], frame)
    endpoints = rotate(layer[:, :6, 3:9].reshape(-1, 6, 2, 3), frame)
    bst = endpoints[..., 1].flatten(1) * 8
    bmt_z = centers[:, BMT_Z, 1] * 8
    bmt_c = centers[:, BMT_C, 2]
    if arm == "g18_endz":
        return torch.cat((bst, bmt_z, bmt_c, endpoints[:, :, 1, 2]), 1), frame

    radius = torch.linalg.vector_norm(layer[..., :2], dim=-1)
    level_c, slope_c = bmt_line(radius[:, BMT_C], bmt_c, present[:, BMT_C])
    if arm == "bmt_cslope_outer16":
        level_z, _ = bmt_line(radius[:, BMT_Z], bmt_z, present[:, BMT_Z])
        summary = torch.stack((level_z, level_c, slope_c, endpoints[:, 5, 1, 2]), 1)
    else:
        summary = torch.cat((chord_inputs(layer, centers, present, frame),
                             torch.stack((level_c, slope_c), 1)), 1)
    return torch.cat((bst, summary), 1), frame
