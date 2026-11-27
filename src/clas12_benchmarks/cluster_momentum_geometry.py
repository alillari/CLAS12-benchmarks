"""Measured cluster aggregation shared by HDF5 preparation and the compact MLP."""

import numpy as np

COMPACT_SHAPE = (12, 10)
GEOMETRY_SHAPE = (12, 12)
GEOMETRY_SCHEMA = "cvt_layer_means_v2"
GEOMETRY_SHAPES = {"cvt_layer_means_v1": COMPACT_SHAPE, GEOMETRY_SCHEMA: GEOMETRY_SHAPE}
MIN_LAYERS = 4
LAYER_FIELDS = ("clusters.detector", "clusters.layer", "clusters.endpoints", "clusters.pitch", "clusters.length")


def aggregate_layers(xyz, endpoints, detector, layer, pitch, length):
    """Return equal-weight cluster means by BST 1–6 then BMT 1–6.

    Each slot stores center xyz, endpoint 1 xyz, endpoint 2 xyz, log1p(count),
    mean BMT-C sweep, and mean log1p(100 * pitch_cm).
    Coordinates are in units of 30 cm. Unoccupied slots are zero.
    """
    if (xyz.shape != (len(layer), 3) or endpoints.shape != (len(layer), 6)
            or detector.shape != layer.shape or layer.ndim != 1
            or pitch.shape != layer.shape or length.shape != layer.shape
            or not np.all(np.isin(detector, (1, 2)) & (layer >= 1) & (layer <= 6))):
        raise ValueError("Invalid cluster layer metadata or geometry shape")
    if not np.isfinite(pitch).all() or not np.all(pitch > 0) or not np.isfinite(length).all():
        raise ValueError("Invalid cluster pitch or length")
    is_arc = (detector == 1) & np.isin(layer, (1, 4, 6))
    radius = np.linalg.norm(xyz[is_arc, :2], axis=1)
    if not np.all(radius > 0):
        raise ValueError("Invalid BMT-C cluster radius")
    sweep = np.zeros(len(layer), dtype=np.float32)
    sweep[is_arc] = length[is_arc] / radius
    values = np.column_stack((xyz / 30, endpoints / 30, sweep, np.log1p(pitch * 100)))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measured cluster geometry")
    slots = layer - 1 + 6 * (detector == 1)
    count = np.bincount(slots, minlength=GEOMETRY_SHAPE[0])
    result = np.empty(GEOMETRY_SHAPE, dtype=np.float32)
    for i, column in zip((*range(9), 10, 11), values.T, strict=True):
        result[:, i] = np.bincount(slots, weights=column, minlength=len(count)) / np.maximum(count, 1)
    result[:, 9] = np.log1p(count)
    return result
