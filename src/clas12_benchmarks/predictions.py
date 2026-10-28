"""Model-independent predictions in original point order; no truth-based repair."""
import json
from pathlib import Path
import numpy as np


def save_predictions(path, event, predictions, point_indices, *, method, cohort_sha256):
    predictions, point_indices = np.asarray(predictions), np.asarray(point_indices)
    n = len(event.raw_features)
    if point_indices.shape != (n,) or not np.issubdtype(point_indices.dtype, np.integer):
        raise ValueError("One integer point identity is required for every original point")
    if not np.array_equal(np.sort(point_indices), np.arange(n)):
        raise ValueError("Duplicate, missing, or unknown point identities")
    if predictions.ndim not in (1, 2) or len(predictions) != n or not np.isfinite(predictions).all():
        raise ValueError("Missing, nonfinite, or misaligned predictions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "clas12_baseline_predictions_v1", "method": method,
                "cohort_sha256": cohort_sha256, "identity": event.identity}
    with path.open("xb") as stream:
        np.savez_compressed(stream, predictions=predictions[np.argsort(point_indices)],
                            original_point_index=np.arange(n), metadata=json.dumps(metadata))
