import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
import shutil

import numpy as np
from mmap_ninja import RaggedMmap

from clas12_benchmarks.data import (Cohort, Event, file_sha256, node_features,
                                     eggnet_features, classification_edges, positive_edges)
from clas12_benchmarks.predictions import save_predictions


class DataPortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = np.array([[1, 0, 0], [0, 2, 1], [-2, 0, -1], [2, 0, 2]], np.float32)
        for name, data in {"features": self.raw, "seg_target": np.array([0, 0, 1, -1]),
                           "noise_target": np.array([1, 1, 1, 0])}.items():
            RaggedMmap.from_generator(self.root / f"{name}_pretrain",
                                      (data.copy() for _ in range(3)), batch_size=3)
        (self.root / "metadata.json").write_text('{"feature_units":"cm"}')
        rows = [{"source_file": "source.hipo", "event": i, "sample_mode": "event"} for i in range(3)]
        (self.root / "samples_pretrain.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.spec = {"schema": "clas12_baseline_cohort_v1", "datasets": {"fixture": {
            "root": str(self.root), "source_split": "pretrain", "input_layout": "xyz",
            "eta_scale": 2.0, "radius_scale": 3.0,
            "metadata_sha256": file_sha256(self.root / "metadata.json"),
            "samples_sha256": file_sha256(self.root / "samples_pretrain.jsonl"),
            "extra_targets": ["noise_target"]}}, "roles": {
                role: [{"dataset": "fixture", "index": i}]
                for i, role in enumerate(("train", "val", "test"))}}
        self.manifest = self.root / "cohort.json"

    def cohort(self):
        self.manifest.write_text(json.dumps(self.spec))
        return Cohort(self.manifest)

    def test_native_ragged_event_and_all_sidecars_stay_aligned(self):
        event = self.cohort().event("test", 0)
        np.testing.assert_array_equal(event.raw_features, self.raw)
        np.testing.assert_array_equal(event.labels["seg_target"], [0, 0, 1, -1])
        self.assertEqual(event.identity["event"], 2)
        self.assertEqual(event.positions.shape, (4, 4))
        np.testing.assert_allclose(event.positions[0], [0, 1, 0, 1/3])
        self.assertAlmostEqual(float(event.positions[1, 0]), float(np.arcsinh(0.5)/2), places=6)
        with self.assertRaisesRegex(ValueError, "energy"):
            node_features(event)
        self.assertEqual(node_features(event, allow_position_only=True).shape, (4, 4))

    def test_source_event_overlap_is_fatal_across_dataset_aliases(self):
        self.spec["datasets"]["alias"] = dict(self.spec["datasets"]["fixture"])
        self.spec["roles"]["test"] = [{"dataset": "alias", "index": 0}]
        with self.assertRaisesRegex(ValueError, "Duplicate source event"):
            self.cohort()

    def test_metadata_drift_is_fatal(self):
        self.spec["datasets"]["fixture"]["metadata_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.cohort()

    def test_point_target_misalignment_is_fatal(self):
        target = self.root / "seg_target_pretrain"
        shutil.rmtree(target)
        RaggedMmap.from_generator(target, (np.array([0, 1]) for _ in range(3)), batch_size=3)
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.cohort().event("train", 0)

    def test_inspect_cli_reads_every_role(self):
        from clas12_benchmarks.__main__ import main
        self.cohort()
        stream = StringIO()
        with patch("sys.argv", ["clas12-benchmarks", "inspect", str(self.manifest)]), redirect_stdout(stream):
            main()
        report = json.loads(stream.getvalue())
        self.assertEqual(report["roles"], {role: {"events": 1, "points": 4} for role in ("train", "val", "test")})

    def test_missing_sidecar_does_not_create_input_directory(self):
        self.spec["datasets"]["fixture"]["extra_targets"] = ["pid_target"]
        with self.assertRaises(FileNotFoundError):
            self.cohort()
        self.assertFalse((self.root / "pid_target_pretrain").exists())

    def test_neighbors_do_not_depend_on_truth_and_exclude_self(self):
        event = self.cohort().event("train", 0)
        before = eggnet_features(event)
        event.labels["seg_target"][:] = -1
        np.testing.assert_array_equal(before, eggnet_features(event))
        edges = classification_edges(event.positions, k=2, radius=100)
        self.assertFalse(np.any(edges[0] == edges[1]))
        self.assertEqual(edges.shape, (2, 8))

    def test_same_track_supervision_never_connects_background(self):
        edges = positive_edges(np.array([0, 0, 1, -1, -1]))
        np.testing.assert_array_equal(edges, [[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "no truncation"):
            positive_edges(np.zeros(5, dtype=int), max_edges=5)

    def test_duplicate_coordinates_do_not_make_self_neighbors(self):
        event = self.cohort().event("train", 0)
        event.positions[:] = event.positions[0]
        self.assertEqual(eggnet_features(event).shape, (4, 12))

    def test_prediction_export_restores_order_and_rejects_duplicates(self):
        cohort = self.cohort()
        event = cohort.event("test", 0)
        output = self.root / "predictions.npz"
        save_predictions(output, event, np.array([30, 10, 40, 20]), [2, 0, 3, 1],
                         method="fixture", cohort_sha256=cohort.sha256)
        with np.load(output, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["predictions"], [10, 20, 30, 40])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            save_predictions(self.root / "invalid.npz", event, [0]*4, [0, 0, 2, 3],
                             method="fixture", cohort_sha256=cohort.sha256)

    def test_eggnet_export_round_trip_has_correct_noise_ids(self):
        from clas12_benchmarks.__main__ import export_eggnet
        import torch

        output = self.root / "eggnet"
        report = export_eggnet(self.cohort(), output, 1000)
        data = torch.load(output / "train/event000000000.pyg", weights_only=False)
        np.testing.assert_array_equal(data.hit_particle_id.numpy(), [1, 1, 2, 0])
        self.assertEqual(len(report["events"]), 3)
        self.assertEqual(data.num_nodes, 4)
        with self.assertRaises(FileExistsError):
            export_eggnet(self.cohort(), output, 1000)


class GraphPortTests(unittest.TestCase):
    def test_all_four_operators_train_and_strictly_reload(self):
        import torch
        from clas12_benchmarks.graphs import GraphClassifier

        torch.set_num_threads(1)
        torch.manual_seed(42)
        x = torch.randn(8, 4)
        edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        for operator in ("GATConv", "GCNConv", "GraphConv", "SAGEConv"):
            with self.subTest(operator=operator):
                model = GraphClassifier(operator, 4, 2)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = model.encoder[0].weight.detach().clone()
                logits = model(x, edges)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, model.encoder[0].weight))
                with tempfile.TemporaryDirectory() as root:
                    checkpoint = Path(root) / "model.pt"
                    torch.save(model.state_dict(), checkpoint)
                    restored = GraphClassifier(operator, 4, 2)
                    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                model.eval()
                restored.eval()
                torch.testing.assert_close(model(x, edges), restored(x, edges), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
