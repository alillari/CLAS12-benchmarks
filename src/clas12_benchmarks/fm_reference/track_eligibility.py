"""Shared CLAS12 finding/noise eligibility, applied before event serialization.

This changes downstream supervision, never physical primary/MC provenance.
"""

import numpy as np

PRIMARY_TRACK_MIN_CLUSTERS = 4
PRIMARY_TRACK_POLICY = "primary_mc_min4_v1"


def primary_track_eligibility_contract():
    """Describe the fixed downstream policy; return an independent metadata copy."""
    return {"policy": PRIMARY_TRACK_POLICY, "min_clusters": PRIMARY_TRACK_MIN_CLUSTERS,
            "count_support": "full_event_owned_clusters", "background_label": -1}


def eligible_primary_labels(labels):
    """Copy a complete event's labels, replacing positive-ID sets of size <4.

    Zero is a valid segment ID. Existing negative labels are preserved; callers
    must not pass cropped/padded events. Surviving IDs are not renumbered.
    """
    original = np.asarray(labels)
    if original.ndim != 1 or original.dtype.kind not in "iu":
        raise ValueError("Expected one-dimensional integer full-event segment labels")
    result = original.astype(np.int64, copy=True)
    ids, counts = np.unique(result[result >= 0], return_counts=True)
    result[np.isin(result, ids[counts < PRIMARY_TRACK_MIN_CLUSTERS])] = -1
    return result


class EligibleTrackLabelView:
    """Lazy full-event projection for legacy ragged arrays."""

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return eligible_primary_labels(self.labels[index])


class EligibleNoiseLabelView:
    """Mark newly excluded track members as noise, preserving other labels."""

    def __init__(self, labels, noise):
        self.labels, self.noise = labels, noise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        truth = np.asarray(self.labels[index])
        noise = np.array(self.noise[index], copy=True)
        if noise.shape != truth.shape:
            raise ValueError("Noise and segmentation labels must align within an event")
        noise[(truth >= 0) & (eligible_primary_labels(truth) < 0)] = 1
        return noise


def configure_primary_track_cut(params):
    """Enable the fixed policy for CLAS12 adapter entry points, not TPC tasks."""
    enabled = getattr(params, "input_layout", "xyz").lower() in {"xyz", "etaphr"}
    setattr(params, "apply_primary_track_cut", enabled)
    if hasattr(params, "params"):
        params.params["apply_primary_track_cut"] = enabled
    if enabled:
        contract = primary_track_eligibility_contract()
        setattr(params, "track_eligibility_contract", contract)
        if hasattr(params, "params"):
            params.params["track_eligibility_contract"] = contract
