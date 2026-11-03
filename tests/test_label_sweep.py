import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from mmap_ninja import RaggedMmap

from clas12_benchmarks.data import Cohort
from clas12_benchmarks.prepare_cohort import prepare
from clas12_benchmarks.prepare_label_sweep import prepare as prepare_sweep


class LabelSweepTests(unittest.TestCase):
    def test_bootstrap_preserves_undefined_metrics(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("label_sweep_runner", Path(__file__).resolve().parents[1] / "scripts/run_label_sweep.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            row = dict(n_matched_tracks=0, n_true_tracks=2, n_pred_tracks=0, ari_signal=0., background_rejection=None)
            path.write_text(json.dumps(row) + "\n")
            result = module.bootstrap_metrics(path, replicates=32)
            self.assertIsNone(result["intervals_95"]["track_purity_global"])
            self.assertIsNone(result["intervals_95"]["background_rejection"])
            self.assertEqual(result["intervals_95"]["track_efficiency_global"], [0., 0.])
    def test_nested_training_fixed_holdouts_and_training_only_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "native"
            root.mkdir()
            (root / "metadata.json").write_text("{}")
            for split in ("pretrain", "test"):
                xyz = [np.array([[1, 0, 1], [0, 2, 2], [3, 0, 3]], np.float32) * (i + 1) for i in range(25)]
                truth = [np.array([0, 0, -1], dtype=np.int64) for _ in xyz]
                for name, values in (("features", xyz), ("seg_target", truth), ("coatjava_seg_pred", truth)):
                    RaggedMmap.from_generator(root / f"{name}_{split}", iter(values), batch_size=25)
                (root / f"samples_{split}.jsonl").write_text("".join(json.dumps({"source_file": split + ".hipo",
                    "event": i, "sample_mode": "event"}) + "\n" for i in range(25)))
            parent = prepare(root, base / "parent", train=8, val=3, test=4, seed=2)
            summary = prepare_sweep(root, base / "parent", base / "sweep", budgets=(2, 4, 8, 16), seed=7)
            previous = set()
            def identities(c, role):
                return [(c.event(role, i).identity["source_file"], c.event(role, i).identity["event"])
                        for i in range(len(c.roles[role]))]
            for budget in (2, 4, 8, 16):
                c = Cohort(base / f"sweep/cohort_{budget}.json")
                train = set(identities(c, "train"))
                self.assertEqual(len(train), budget)
                self.assertTrue(previous <= train)
                previous = train
                self.assertEqual(identities(c, "val"), identities(parent, "val"))
                self.assertEqual(identities(c, "test"), identities(parent, "test"))
                xyz = np.concatenate([c.event("train", i).raw_features for i in range(budget)])
                r = np.hypot(xyz[:, 0], xyz[:, 1])
                self.assertEqual(summary["budgets"][str(budget)]["normalization"]["radius_scale"], float(r.max()))
                if budget == 8:
                    self.assertEqual(train, set(identities(parent, "train")))
                    for i in range(budget):
                        np.testing.assert_array_equal(c.event("train", i).positions, parent.event("train", i).positions)


if __name__ == "__main__":
    unittest.main()
