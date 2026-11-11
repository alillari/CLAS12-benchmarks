import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.fm_reference.noise_metrics import at_threshold, calibrate_thresholds, score_summary
from clas12_benchmarks.graphs import OPERATORS
from clas12_benchmarks.hdf5 import ClusterSource, MIN4_NOISE_POLICY
from clas12_benchmarks.noise_tagging import main, max_balanced_accuracy_threshold

from hdf5_fixture import make_publication


class NoiseTaggingTests(unittest.TestCase):
    def test_exact_balanced_accuracy_sweep_with_ties(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0., 0., .4, .4, .8, 1.], dtype=np.float32)
        thresholds = [-np.finfo(np.float32).eps, *np.unique(scores)]
        expected = max(thresholds, key=lambda t: at_threshold(labels, scores, t)["balanced_accuracy"])
        self.assertEqual(max_balanced_accuracy_threshold(labels, scores), expected)
        np.testing.assert_equal(max_balanced_accuracy_threshold(labels[:2], scores[:2]), thresholds[0])

    def test_cli_imports_fm_members_trains_and_scores_all_operators(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clusters = dict(
                xyz=np.column_stack((np.arange(9) + 1, np.ones(9), np.arange(9))).astype("f4"),
                primary_segment=np.array([3] * 4 + [9] * 3 + [-1] * 2, dtype="i4"),
                primary_tid=np.array([17] * 4 + [42] * 3 + [-1] * 2, dtype="i4"),
                reco_segment=np.array([0] * 4 + [1] * 3 + [-1] * 2, dtype="i4"),
                primary_pid=np.full(9, 211, dtype="i4"), primary_pid_valid=np.ones(9, bool),
                primary_noise=np.array([0] * 7 + [1] * 2, dtype="i1"),
                primary_noise_valid=np.ones(9, bool),
            )
            data = make_publication(root / "data", cluster_fields=clusters)
            rows = {}
            for split in ("train", "validation", "test"):
                with ClusterSource(data, split) as source:
                    identities = source.identities([3, 1])
                rows[split] = [dict(source_event_index=index, source_id=identities[index]["source_id"],
                                    original_ordinal=identities[index]["original_ordinal"], segment_label=None)
                               for index in (3, 1)]
            fm = root / "cohorts.json"
            fm.write_text(json.dumps(rows))
            output = root / "run"
            args = ["noise_tagging", "--data-root", str(data), "--fm-cohorts", str(fm), "--plan-id", "fixture-plan",
                    "--output", str(output), "--steps", "2", "--val-interval", "1", "--batch-size", "2",
                    "--class-balance", "inverse_frequency"]
            with patch("sys.argv", args), patch("builtins.print"):
                main()
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(fm.read_bytes(), (output / "cohorts.json").read_bytes())
            with Cohort(output / "cohort.json") as cohort:
                for role in ("train", "val", "test"):
                    self.assertEqual([member["index"] for member in cohort.roles[role]], [3, 1])
                    event = cohort.event(role, 0)
                    np.testing.assert_array_equal(event.labels["seg_target"], [3] * 4 + [9] * 3 + [-1] * 2)
                    np.testing.assert_array_equal(event.labels["noise_target"], [0] * 4 + [1] * 5)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(set(report["models"]), set(OPERATORS))
            for operator, relative in report["models"].items():
                with self.subTest(operator=operator):
                    folder = output / operator
                    result = json.loads((output / relative).read_text())
                    history = json.loads((folder / "training_history.json").read_text())
                    self.assertEqual([entry["step"] for entry in history], [1, 2])
                    best = min(history, key=lambda entry: entry["validation_loss"])
                    self.assertEqual(result["selected_checkpoint_step"], best["step"])
                    self.assertEqual(result["selected_validation_loss"], best["validation_loss"])
                    self.assertEqual(result["data_contract"]["noise_policy"], MIN4_NOISE_POLICY)
                    self.assertEqual(result["training_class_counts"], [8, 10])
                    self.assertEqual(result["checkpoint_sha256"], file_sha256(folder / "checkpoint.pt"))
                    ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(ckpt["operator"], operator)
                    with np.load(folder / "validation_scores.npz") as val, np.load(folder / "test_scores.npz") as test:
                        np.testing.assert_array_equal(test["event_offsets"], [0, 9, 18])
                        np.testing.assert_array_equal(test["labels"], ([0] * 4 + [1] * 5) * 2)
                        np.testing.assert_array_equal(test["coatjava_noise"], ([0] * 7 + [1] * 2) * 2)
                        fm_calibration = calibrate_thresholds(val["labels"], val["noise_score"], val["coatjava_noise"])
                        for name, point in fm_calibration.items():
                            scored = at_threshold(test["labels"], test["noise_score"], point["threshold"])
                            self.assertEqual(result["operating_points"][name],
                                             {**scored, "target_signal_retention": point["target_signal_retention"]})
                        expected = score_summary(test["labels"], test["noise_score"], test["coatjava_noise"])
                        self.assertEqual(result["coatjava_unassigned"], expected["coatjava_unassigned"])
                        self.assertEqual(result["adapter"], expected["adapter"])
