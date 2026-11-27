"""Measured cluster aggregation shared by HDF5 preparation and the compact MLP."""

import numpy as np

GEOMETRY_SHAPE = (12, 10)
MIN_LAYERS = 4
LAYER_FIELDS = ("clusters.detector", "clusters.layer", "clusters.endpoints")


def aggregate_layers(xyz, endpoints, detector, layer):
    """Return equal-weight cluster means by BST 1–6 then BMT 1–6.

    Each slot stores center xyz, endpoint 1 xyz, endpoint 2 xyz, and log1p(count).
    Coordinates are in units of 30 cm. Unoccupied slots are zero.
    """
    if (xyz.shape != (len(layer), 3) or endpoints.shape != (len(layer), 6)
            or detector.shape != layer.shape or layer.ndim != 1
            or not np.all(np.isin(detector, (1, 2)) & (layer >= 1) & (layer <= 6))):
        raise ValueError("Invalid cluster layer metadata or geometry shape")
    values = np.column_stack((xyz / 30, endpoints / 30))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite measured cluster geometry")
    slots = layer - 1 + 6 * (detector == 1)
    count = np.bincount(slots, minlength=GEOMETRY_SHAPE[0])
    result = np.empty(GEOMETRY_SHAPE, dtype=np.float32)
    for i, column in enumerate(values.T):
        result[:, i] = np.bincount(slots, weights=column, minlength=len(count)) / np.maximum(count, 1)
    result[:, 9] = np.log1p(count)
    return result
