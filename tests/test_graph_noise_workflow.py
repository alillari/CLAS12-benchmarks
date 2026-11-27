"""Exercise the v2 cache, validation-only selection and one held-out scoring pass."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.graph_noise import run
from clas12_benchmarks.graph_noise_data import freeze
from clas12_benchmarks.hdf5 import FORMAT_V2

from hdf5_fixture import make_publication


class GraphNoiseWorkflowTests(unittest.TestCase):
    def test_v2_graph_noise_outputs_share_the_roster(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)
        n = 9
        clusters = dict(
            xyz=np.column_stack((np.arange(n) + 1, np.ones(n), np.arange(n))).astype("f4"),
            detector=np.array([1] * 4 + [2] * 5, dtype="i4"),
            layer=np.ones(n, dtype="i4"),
            primary_segment=np.array([3] * 4 + [9] * 3 + [-1] * 2, dtype="i4"),
            primary_tid=np.array([17] * 4 + [42] * 3 + [-1] * 2, dtype="i4"),
            reco_segment=np.array([0] * 4 + [1] * 3 + [-1] * 2, dtype="i4"),
            primary_pid=np.full(n, 211, dtype="i4"),
            primary_pid_valid=np.ones(n, dtype=bool),
            primary_noise=np.array([0] * 7 + [1] * 2, dtype="i1"),
            primary_noise_valid=np.ones(n, dtype=bool),
        )
        root = make_publication(tmp_path / "data", cluster_fields=clusters, storage_format=FORMAT_V2)
        cache = tmp_path / "cache"
        freeze(root, cache, (3, 3, 3), 8)
        arm = {"name": "grav_small", "lr": .001, "class_balance": "inverse_frequency",
               "model": dict(architecture="gravnet", width=8, layers=1, propagate=4, neighbors=3)}
        settings = dict(train=3, validation=3, steps=2, val_interval=1, batch_size=1, seed=71)
        config = {"arms": [arm], "search": dict(**settings, arms=[arm["name"]], test=False),
                  "final": dict(**settings, test=True)}
        search = tmp_path / "search"
        run(cache, config, "search", search, "cpu")
        selected = json.loads((search / "summary.json").read_text())
        self.assertNotIn("test", selected["arms"][arm["name"]])
        result = tmp_path / "result"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            run(cache, config, "final", result, "cpu")
        run(cache, config, "final", result, "cpu", selected)
        report = json.loads((result / "summary.json").read_text())
        self.assertEqual(report["cohort"]["splits"]["train"]["contract"]["format"], FORMAT_V2)
        test = report["arms"][arm["name"]]["test"]
        self.assertEqual(test["signal_clusters"], 12)
        self.assertEqual(test["noise_clusters"], 15)
        self.assertIn("retain_99p5pct", report["bootstrap"])
        self.assertEqual(report["provenance_sha256"], file_sha256(result / "provenance.json"))
        table = result / arm["name"] / "test_scores.npz"
        with np.load(table) as scores, np.load(cache / "test.npz") as original:
            for a, b in [("event_offsets", "offsets"), ("source_event_index", "source_event_index"),
                         ("detector", "detector"), ("labels", "labels")]:
                np.testing.assert_array_equal(scores[a], original[b])
        detail = json.loads((result / arm["name"] / "test.json").read_text())
        self.assertIn("BMT_layer_1", detail["groups"])
