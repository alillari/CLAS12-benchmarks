"""Evaluate an existing frozen pipeline on a new, training-compatible cohort."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .data import Cohort, file_sha256, event_key
from .sequences import derived_sequence
from .prepare_cohort import write_json
from .tracking import (metric_module, embedding_output, candidates, filter_probabilities,
                       graph_predictions, evaluate, eggnet_prediction)
from .upstream import eggnet, exatrkx_embedding, exatrkx_filter, ROOT
from .interaction_network import SegmentClassifier
from .finding_evaluation import FindingEvaluation, finding_sequence, finding_fingerprints, verify_reference_provenance


def evaluate_frozen(parent, original_cohort, cohort_path, output, device="cuda", threads=4):
    parent, output = Path(parent), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if output.resolve().is_relative_to(parent.resolve()):
        raise ValueError("Frozen output must be outside the parent run")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "provenance.json").exists():
        raise FileExistsError(output)
    provenance = json.loads((parent / "provenance.json").read_text())
    verify_reference_provenance(provenance)
    with Cohort(original_cohort) as old, Cohort(cohort_path) as cohort:
        if old.sha256 != provenance["cohort_sha256"]:
            raise ValueError("Wrong original cohort")
        if len(old.roles["train"]) != len(cohort.roles["train"]):
            raise ValueError("Training population changed")
        for i in range(len(old.roles["train"])):
            a, b = old.event("train", i), cohort.event("train", i)
            if event_key(a.identity) != event_key(b.identity):
                raise ValueError("Training identities/order changed")
            np.testing.assert_array_equal(a.positions, b.positions)
            np.testing.assert_array_equal(a.raw_features, b.raw_features)
            np.testing.assert_array_equal(a.labels["seg_target"], b.labels["seg_target"])
        for name in ("vendor/eggnet/eggnet/models/eggnet.py", "vendor/eggnet/eggnet/models/utils/utils.py",
                     "vendor/exatrkx/exatrkx/src/embedding/layerless_embedding.py", "vendor/exatrkx/exatrkx/src/filter/vanilla_filter.py",
                     "src/clas12_benchmarks/interaction_network.py"):
            if file_sha256(ROOT / name) != provenance["source_sha256"][name]:
                raise ValueError(f"Frozen architecture changed: {name}")
        metrics = metric_module(parent / "track_finding_metrics.py", output)
        if file_sha256(output / "track_finding_metrics.py") != provenance["metric_sha256"]:
            raise ValueError("Frozen evaluator changed")
        selected = json.loads((parent / "selected_checkpoints.json").read_text())
        for row in selected:
            path = parent / f"{row['stage']}_best.pt"
            if file_sha256(path) != row["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
            shutil.copyfile(path, output / path.name)
        shutil.copyfile(parent / "selected_checkpoints.json", output / "selected_checkpoints.json")
        shutil.copyfile(parent / "frozen_selection.json", output / "frozen_selection.json")
        config, method = provenance["config"], provenance["method"]
        evaluation = FindingEvaluation(metrics, provenance.get("finding_evaluation"))
        if "finding_evaluation" in provenance:
            if config.get("finding_evaluation") != evaluation.contract:
                raise ValueError("Training/evaluation contract mismatch")
            if finding_fingerprints(old, evaluation) != provenance.get("finding_data_sha256"):
                raise ValueError("Original finding provenance sidecars changed")
            provenance["finding_data_sha256"] = finding_fingerprints(cohort, evaluation)
        provenance.update(training_git_commit=provenance["git_commit"], git_commit=os.environ.get("BENCHMARK_GIT_COMMIT"),
            original_cohort_sha256=old.sha256, cohort_sha256=cohort.sha256, evaluation_only=True,
            parent_run=str(parent), thresholds_recalibrated=False, preserved_training_verified=True,
            evaluation_source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in (ROOT / "src").rglob("*.py")})
        provenance["data_array_sha256"] = cohort.data_fingerprints()
        write_json(output / "provenance.json", provenance)
        torch.set_num_threads(threads)
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        def load(model, name):
            model.load_state_dict(torch.load(parent / f"{name}_best.pt", map_location="cpu", weights_only=False)["model"], strict=True)
            return model.to(device).eval()
        selection = json.loads((parent / "frozen_selection.json").read_text())
        test = finding_sequence(cohort, "test", evaluation)
        with torch.no_grad():
            if method == "eggnet":
                model = load(eggnet(config["eggnet"]), "embedding")
                predictions = derived_sequence(test, lambda i: eggnet_prediction(model, test[i], selection, device))
            else:
                ec = config["exatrkx"]
                model = load(exatrkx_embedding(ec["embedding"]), "embedding")
                filt = load(exatrkx_filter(ec["filter"]), "filter")
                gnn = load(SegmentClassifier(4, ec["gnn_processing_steps"]), "gnn")
                filter_selection = json.loads((parent / "filter_selection.json").read_text())
                write_json(output / "filter_selection.json", filter_selection)
                cut = filter_selection["selected"]["cut"]
                raw_graphs = derived_sequence(test, lambda i: candidates(model, test[i], ec, device))
                graphs = derived_sequence(test, lambda i: raw_graphs[i][:,
                    filter_probabilities(filt, test[i], raw_graphs[i], device) >= cut])
                predictions = graph_predictions(gnn, test, graphs, selection["gnn_cut"], ec, device)
        result = evaluate(test, predictions, metrics, output, method, cohort.sha256, provenance.get("finding_evaluation"))
        print("FROZEN_EVALUATION_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--original-cohort", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    evaluate_frozen(args.parent, args.original_cohort, args.cohort, args.output, args.device, args.threads)
