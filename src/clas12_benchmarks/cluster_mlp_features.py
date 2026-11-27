"""Measured cluster coordinates and derived inputs for the momentum recipes."""

import torch

BMT_Z = [7, 8, 10]
BMT_C = [6, 9, 11]

# Column order in the archived study encoder and checkpoint files.
BST_END_Y = [19 * layer + endpoint for layer in range(6) for endpoint in (4, 7)]
XYZ_COLUMNS = [19 * layer + axis for layer in range(12) for axis in range(9)]
# The 218-input control keeps the first representative of each nonconstant
# column in audit order, including coefficients that received folded duplicates.
DISTINCT_COLUMNS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 270, 285, 10, 15, 16, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 29, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    48, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 86, 91, 92, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 105, 110, 111, 113, 114, 115, 116, 117, 118, 120, 121, 123,
    124, 129, 130, 132, 133, 134, 135, 138, 141, 143, 148, 149, 151, 152, 153,
    154, 157, 160, 162, 167, 168, 170, 171, 172, 173, 174, 175, 177, 178, 180,
    181, 186, 187, 189, 190, 191, 192, 195, 198, 200, 205, 206, 208, 209, 210,
    211, 212, 213, 215, 216, 218, 219, 224, 225, 227,
    *range(228, 270), 271, 272, *range(273, 285), *range(286, 302), 303, 305, 306,
]
LAB_INPUT = {"xyz108_lab", "xyz108_lab_local_target", "xyz108_lab_circle"}
LAB_TARGET = {"xyz108_lab", "xyz108_lab_circle"}
CIRCULAR = {"xyz108_local_circle", "xyz108_lab_circle"}
XYZ_ARMS = ("xyz108_local", "xyz108_lab", "xyz108_lab_local_target",
            "xyz108_local_circle", "xyz108_lab_circle")
ARMS = {
    "g18_endz": BST_END_Y + [19 * layer + 1 for layer in BMT_Z]
    + [19 * layer + 2 for layer in BMT_C] + [19 * layer + 8 for layer in range(6)],
    "bmt_cslope_outer16": BST_END_Y + [320, 322, 323, 19 * 5 + 8],
    "bst_angle17": BST_END_Y + [326, 324, 325, 322, 323],
    "distinct218": DISTINCT_COLUMNS,
    **dict.fromkeys(XYZ_ARMS, XYZ_COLUMNS),
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
    return midpoint, distance, valid


def chord_inputs(layer, centers, present, frame):
    stereo, _, stereo_valid = stereo_points(layer, present, frame)
    stereo = stereo.masked_fill(~stereo_valid[..., None], 0.)
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
    """Encode measured layer geometry and return (inputs, frame)."""
    if arm not in ARMS:
        raise ValueError(f"Unsupported cluster MLP recipe: {arm}")
    present = layer[..., 9] > 0
    transverse = present.clone()
    transverse[:, BMT_C] = False
    anchor = (layer[..., :2] * (transverse | ~transverse.any(1, keepdim=True))[..., None]).sum(1)
    alpha = torch.atan2(anchor[:, 1], anchor[:, 0])
    frame = torch.stack((alpha.cos(), alpha.sin()), 1)
    if arm in XYZ_ARMS:
        coordinates = layer[..., :9].reshape(-1, 12, 3, 3)
        if arm not in LAB_INPUT:
            coordinates = rotate(coordinates, frame)
        if arm in LAB_TARGET:
            frame = torch.zeros_like(frame)
            frame[:, 0] = 1.
        scale = coordinates.new_tensor([1., 8., 1.])
        return (coordinates * scale).flatten(1), frame
    if arm == "distinct218":
        return encode_distinct(layer, present, transverse, frame), frame
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


def line_summary(x, y, mask):
    weight = mask.float()
    count = weight.sum(1, keepdim=True)
    mean_x = (x * weight).sum(1, keepdim=True) / count.clamp_min(1)
    mean_y = (y * weight).sum(1, keepdim=True) / count.clamp_min(1)
    dx = x - mean_x
    variance = (dx.square() * weight).sum(1, keepdim=True)
    slope = ((dx * (y - mean_y) * weight).sum(1, keepdim=True) / variance.clamp_min(1e-5)).clamp(-4, 4)
    scatter = ((((dx * slope + mean_y - y).square() * weight)
                .sum(1, keepdim=True)) / count.clamp_min(1)).sqrt()
    return torch.cat((count, mean_x, mean_y, slope, scatter), 1)


def pair_bending(points, mask, z_points, z_mask):
    first, second = torch.triu_indices(points.size(1), points.size(1), 1, device=points.device)
    a, b = points[:, first], points[:, second]
    r = torch.linalg.vector_norm(points, dim=-1)
    ra, rb = r[:, first], r[:, second]
    chord = torch.linalg.vector_norm(b - a, dim=-1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    delta = rb - ra
    valid = mask[:, first] & mask[:, second] & (delta.abs() > .09)
    curvature = (2 * cross * delta.sign() / (ra * rb * chord).clamp_min(1e-6)).clamp(-2, 2)
    curvature = curvature.masked_fill(~valid, 0)
    weight = delta.square() * valid
    total = weight.sum(1, keepdim=True).clamp_min(1e-6)
    mean = (curvature * weight).sum(1, keepdim=True) / total
    spread = (((curvature - mean).square() * weight).sum(1, keepdim=True) / total).sqrt()
    phi = torch.atan2(points[..., 1], points[..., 0]) - (r * mean * .5).clamp(-.99, .99).asin()
    phase = torch.cat((phi.sin() * mask, phi.cos() * mask), 1)
    z_radius = torch.linalg.vector_norm(z_points[..., :2], dim=-1)
    k = mean.abs()
    arc = (z_radius * k * .5).clamp_max(.99).asin() * 2 / k.clamp_min(1e-6)
    arc = torch.where(k > 1e-6, arc, z_radius)
    longitudinal = line_summary(arc, z_points[..., 2], z_mask)
    return torch.cat((curvature, valid, phase, mean, spread, longitudinal), 1)


def encode_distinct(layer, present, transverse, frame):
    """Select the archived 218 ordered columns from the original encoding."""
    if layer.shape[-1] != 12:
        raise ValueError("distinct218 needs measured pitch and BMT-C sweep in the geometry")
    coordinates = rotate(layer[..., :9].reshape(-1, 12, 3, 3), frame)
    xyz = coordinates[:, :, 0]
    scale = coordinates.new_tensor([1., 8., 1.])
    # Retain original column positions, including the historical zero padding.
    slots = torch.cat(((coordinates * scale).flatten(2), layer[..., 10:12],
                       layer.new_zeros((*layer.shape[:2], 4)), layer[..., :3], layer[..., 9:10]), 2)
    stereo, distance, stereo_mask = stereo_points(layer, present, frame)
    stereo_flat = torch.cat((stereo * scale, distance[..., None]), 2).masked_fill(~stereo_mask[..., None], 0)

    r = xyz[..., :2].norm(dim=2)
    rho = stereo[..., :2].norm(dim=2)
    phi = torch.atan2(xyz[..., 1], xyz[..., 0])
    psi = torch.atan2(stereo[..., 1], stereo[..., 0])
    xy_mask = torch.cat((stereo_mask, present[:, BMT_Z]), 1)
    z_mask = torch.cat((stereo_mask, present[:, BMT_C]), 1)
    xy_fit = line_summary(torch.cat((rho, r[:, BMT_Z]), 1), torch.cat((psi, phi[:, BMT_Z]), 1), xy_mask)
    wide_fit = line_summary(r, phi, transverse)
    z_fit = line_summary(torch.cat((rho, r[:, BMT_C]), 1),
                         torch.cat((stereo[..., 2], xyz[:, BMT_C, 2]), 1), z_mask)
    pairs = pair_bending(torch.cat((stereo[..., :2], xyz[:, BMT_Z, :2]), 1), xy_mask,
                         torch.cat((stereo, xyz[:, BMT_C]), 1), z_mask)
    original = torch.cat((slots.flatten(1), stereo_flat.flatten(1), stereo_mask.float(),
                          xy_fit, wide_fit, z_fit, pairs), 1)
    return original[:, DISTINCT_COLUMNS]
