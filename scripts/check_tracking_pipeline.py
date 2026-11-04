"""Exercise training, validation selection and final export on synthetic events.

Pass the shared CLAS12 metric module explicitly; no CLAS12 trainer is imported.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mmap_ninja import RaggedMmap

from clas12_benchmarks.data import Cohort
from clas12_benchmarks.prepare_cohort import prepare, write_json
from clas12_benchmarks.tracking import run
from clas12_benchmarks.upstream import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=str(ROOT / "configs/tracking.json"))
    args = parser.parse_args()
    output = Path(args.output)
    root = output / "fixture"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text("{}")
    rng = np.random.default_rng(7)
    for split in ("pretrain", "test"):
        features, targets = [], []
        for _ in range(12):
            features.append(np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [-1, 0, 1], [-2, 0, 2],
                                      [-3, 0, 3], [0, 2, -1]], np.float32) + rng.normal(0, .01, (7, 3)).astype(np.float32))
            targets.append(np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int64))
        for name, values in (("features", features), ("seg_target", targets), ("coatjava_seg_pred", targets)):
            RaggedMmap.from_generator(root / f"{name}_{split}", iter(values), batch_size=12)
        (root / f"samples_{split}.jsonl").write_text("".join(json.dumps({"source_file": split + ".hipo",
            "event": i, "sample_mode": "event"}) + "\n" for i in range(12)))
    prepare(root, output / "cohort", train=5, val=3, test=3)
    config = json.loads(Path(args.config).read_text())
    config.update(steps_per_stage=4, validation_interval=2, checkpoint_validation_events=3)
    if "warmup_steps" in config:
        config["warmup_steps"] = 1
    config["eggnet"].update(dbscan_eps=[.1, .5], dbscan_min_samples=[2])
    config["exatrkx"]["gnn_cuts"] = [.3, .7]
    write_json(output / "config.json", config)
    cohort = Cohort(output / "cohort/cohort.json")
    for method in ("eggnet", "exatrkx"):
        run(SimpleNamespace(output=str(output / method), config=str(output / "config.json"),
            cohort=str(output / "cohort/cohort.json"), method=method, metrics=args.metrics,
            device=args.device, threads=2, smoke=False, resume=False))
        prediction_paths = sorted((output / method / "predictions").glob("*.npz"))
        assert len(prediction_paths) == 3
        for i, path in enumerate(prediction_paths):
            with np.load(path) as prediction:
                np.testing.assert_array_equal(prediction["original_point_index"], cohort.event("test", i).original_point_index)
                assert len(prediction["predictions"]) == 7
        summary = json.loads((output / method / "summary.json").read_text())
        assert summary["metrics"]["n_events"] == 3
        assert summary["coatjava"]["track_efficiency_global"] == 1.
    print("PIPELINE_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
