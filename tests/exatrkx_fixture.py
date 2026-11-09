"""Small real readers/checkpoints with a deliberately simple frozen evaluator.

The test evaluator counts exact track matches only; it is not a replacement for
the external CLAS12 physics evaluator. Zeroed model weights make edge scores
exactly 0.5, so expected partitions do not depend on the inference implementation.
"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmap_ninja import RaggedMmap

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.interaction_network import SegmentClassifier
from clas12_benchmarks.prepare_cohort import prepare, write_json
from clas12_benchmarks.tracking import metric_module
from clas12_benchmarks.upstream import ROOT, exatrkx_embedding, exatrkx_filter


METRICS = '''import numpy as np
from sklearn.metrics import adjusted_rand_score

def divide(a, b):
    return a / b if b else None

def event_track_metrics(truth, prediction):
    true_ids = np.unique(truth[truth >= 0])
    pred_ids = np.unique(prediction[prediction >= 0])
    matched = sum(any(np.array_equal(truth == t, prediction == p) for p in pred_ids) for t in true_ids)
    signal = truth >= 0
    return {"n_true_tracks": len(true_ids), "n_pred_tracks": len(pred_ids),
            "n_matched_tracks": int(matched),
            "ari_signal": float(adjusted_rand_score(truth[signal], prediction[signal])) if signal.any() else None,
            "merge_rate": divide(sum(len(np.unique(truth[(prediction == p) & signal])) > 1 for p in pred_ids), len(pred_ids)),
            "signal_loss_to_background": divide(int((signal & (prediction < 0)).sum()), int(signal.sum()))}

def summarize_event_metrics(rows):
    out = {k: sum(row[k] for row in rows) for k in ("n_true_tracks", "n_pred_tracks", "n_matched_tracks")}
    out["n_events"] = len(rows)
    for key in ("ari_signal", "merge_rate", "signal_loss_to_background"):
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    out["track_purity_global"] = divide(out["n_matched_tracks"], out["n_pred_tracks"])
    out["track_efficiency_global"] = divide(out["n_matched_tracks"], out["n_true_tracks"])
    return out
'''


def frozen_run(root, backend="materialized", cut=.3):
    root = Path(root)
    root.mkdir(parents=True)
    raw = root / "raw"
    if backend == "hdf5":
        from hdf5_fixture import make_publication
        make_publication(raw)
        manifest = root / "cohort/cohort.json"
        with redirect_stdout(StringIO()), prepare(raw, manifest.parent, reader="hdf5", train=1, val=1, test=1):
            pass
    else:
        raw.mkdir()
        xyz = np.array([[1, 0, 1], [2, 0, 2], [0, 1, 1], [0, 2, 2], [-1, 0, 1]], np.float32)
        for field, values in (("features", xyz), ("seg_target", np.array([0, 0, 1, 1, -1], np.int64))):
            RaggedMmap.from_generator(raw / f"{field}_pretrain", (values.copy() for _ in range(3)), batch_size=3)
        (raw / "metadata.json").write_text('{}')
        samples = raw / "samples_pretrain.jsonl"
        samples.write_text(''.join(json.dumps({"source_file": "fixture.hipo", "event": i, "sample_mode": "event"}) + '\n' for i in range(3)))
        manifest = (raw if backend == "materialized" else root / "cohort") / "cohort.json"
        manifest.parent.mkdir(exist_ok=True)
        write_json(manifest, {"schema": "clas12_baseline_cohort_v1", "datasets": {"source": {
            "root": str(raw), "source_split": "pretrain", "input_layout": "xyz", "eta_scale": 1., "radius_scale": 2.,
            "metadata_sha256": file_sha256(raw / "metadata.json"), "samples_sha256": file_sha256(samples)}},
            "roles": {role: [{"dataset": "source", "index": i}] for i, role in enumerate(("train", "val", "test"))}})
    parent = root / "run"
    parent.mkdir()
    evaluator = root / "fixture_metrics.py"
    evaluator.write_text(METRICS)
    metrics = metric_module(evaluator, parent)
    config = json.loads((ROOT / "configs/tracking.json").read_text())["exatrkx"]
    config["embedding"].update(emb_hidden=8, nb_layer=2, emb_dim=2)
    config["filter"].update(hidden=8, nb_layer=2)
    config.update(gnn_processing_steps=1, candidate_k=32, candidate_radius=2.)
    models = {"embedding": exatrkx_embedding(config["embedding"]), "filter": exatrkx_filter(config["filter"]),
              "gnn": SegmentClassifier(4, 1)}
    selected = []
    for name, model in models.items():
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        path = parent / f"{name}_best.pt"
        torch.save({"model": model.state_dict(), "step": 1}, path)
        selected.append({"stage": name, "selected_step": 1, "checkpoint_sha256": file_sha256(path)})
    write_json(parent / "selected_checkpoints.json", selected)
    write_json(parent / "filter_selection.json", {"selected": {"cut": .3}})
    with Cohort(manifest) as cohort:
        truth = cohort.event("val", 0).labels["seg_target"]
        expected = np.full(len(truth), 0 if cut <= .5 else -1, dtype=np.int64)
        write_json(parent / "frozen_selection.json", {"gnn_cut": cut, "metrics": metrics.summarize_event_metrics([metrics.event_track_metrics(truth, expected)])})
        _, sizes = np.unique(truth[truth >= 0], return_counts=True)
        write_json(parent / "candidate_diagnostics.json", {"val": {"true_pairs": int(np.sum(sizes * (sizes - 1) // 2))}})
        sources = [p for folder in ("src", "vendor") for p in (ROOT / folder).rglob("*.py")]
        write_json(parent / "provenance.json", {"method": "exatrkx", "cohort_sha256": cohort.sha256,
            "data_array_sha256": cohort.data_fingerprints(), "config": {"exatrkx": config},
            "metric_sha256": file_sha256(parent / "track_finding_metrics.py"),
            "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}})
    return SimpleNamespace(run=str(parent), cohort=str(manifest), output=str(root / "diagnostics"),
        split="val", max_events=None, device="cpu", threads=2, radii=[0., 2., float("inf")], ks=[1, 32, None],
        cuts=[0., .3, .7, 1.], bootstrap=8)
