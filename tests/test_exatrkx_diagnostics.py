import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from clas12_benchmarks.exatrkx_diagnostics import (aggregate_graphs, graph_diagnostics,
                                                  mst_radius, undirected_candidates,
                                                  validate_selected_stages)
from clas12_benchmarks import exatrkx_diagnostics
from exatrkx_fixture import frozen_run


class DiagnosticIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def run_cli(self, args):
        argv = ["diagnostics", "--run", args.run, "--cohort", args.cohort, "--output", args.output,
                "--device", "cpu", "--threads", "2", "--radii", "0", "2", "inf", "--ks", "1", "32", "all"]
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            exatrkx_diagnostics.main()
        return json.loads((Path(args.output) / "summary.json").read_text())

    def test_cli_accepts_all_storage_fingerprints_and_replays_baseline(self):
        for backend in ("materialized", "external", "hdf5"):
            with self.subTest(backend=backend):
                args = frozen_run(self.root / backend, backend)
                report = self.run_cli(args)
                self.assertEqual(report["events"], 1)
                self.assertEqual(report["replay_minus_saved_validation"], {
                    "ari_signal": 0., "track_purity_global": 0., "track_efficiency_global": 0.})
                self.assertEqual(report["stages"]["candidate"]["graph"]["true_pair_recall"], 1.)

    def test_cli_rejects_changed_external_payload(self):
        from mmap_ninja import RaggedMmap
        args = frozen_run(self.root / "external", "external")
        array = RaggedMmap(self.root / "external/raw/features_pretrain", mode="r+")
        array[1][:] *= 2
        with self.assertRaisesRegex(ValueError, "Prepared data fingerprints changed"):
            self.run_cli(args)

    def test_cli_rejects_changed_hdf5_payload(self):
        import h5py
        args = frozen_run(self.root / "hdf5", "hdf5")
        for path in (self.root / "hdf5/raw/validation").glob("*.h5"):
            with h5py.File(path, "r+") as source:
                source["clusters/xyz"][:] *= 2
        with self.assertRaisesRegex(ValueError, "content drift"):
            self.run_cli(args)

    def test_cli_rejects_missing_or_changed_inference_sources_before_loading(self):
        args = frozen_run(self.root / "source-check")
        path = Path(args.run) / "provenance.json"
        original = json.loads(path.read_text())
        for source in ("tracking.py", "data.py", "graphs.py", "sources.py", "sequences.py"):
            for value in (None, "0" * 64):
                with self.subTest(source=source, value=value):
                    provenance = json.loads(json.dumps(original))
                    key = "src/clas12_benchmarks/" + source
                    if value is None:
                        del provenance["source_sha256"][key]
                    else:
                        provenance["source_sha256"][key] = value
                    path.write_text(json.dumps(provenance))
                    with patch.object(exatrkx_diagnostics, "exatrkx_embedding", side_effect=AssertionError("Must reject before model loading")):
                        with self.assertRaisesRegex(ValueError, "source changed or missing.*" + source):
                            self.run_cli(args)
                    self.assertFalse((Path(args.output) / "provenance.json").exists())

    def test_cli_rejects_changed_hdf5_reader_source(self):
        args = frozen_run(self.root / "hdf5-source", "hdf5")
        path = Path(args.run) / "provenance.json"
        provenance = json.loads(path.read_text())
        provenance["source_sha256"]["src/clas12_benchmarks/hdf5.py"] = "0" * 64
        path.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, "source changed or missing.*hdf5.py"):
            self.run_cli(args)

    def test_no_predictions_produce_null_replay_delta_and_renderable_outputs(self):
        args = frozen_run(self.root / "empty", cut=1.)
        report = self.run_cli(args)
        self.assertEqual(report["actual_gnn"]["n_pred_tracks"], 0)
        self.assertIsNone(report["actual_gnn"]["track_purity_global"])
        self.assertIsNone(report["replay_minus_saved_validation"]["track_purity_global"])
        self.assertEqual(report["replay_minus_saved_validation"]["track_efficiency_global"], 0.)
        script = Path(__file__).resolve().parents[1] / "scripts/plot_exatrkx_diagnostics.py"
        subprocess.run([sys.executable, str(script), "--input", args.output], check=True, capture_output=True, text=True)
        for name in ("embedding_and_candidates", "oracle_reconstruction"):
            for suffix in ("png", "svg"):
                self.assertGreater((Path(args.output) / f"{name}.{suffix}").stat().st_size, 0)

    def test_replay_delta_is_null_when_only_one_side_is_undefined(self):
        for cut, saved_purity in ((.3, None), (1., 0.)):
            with self.subTest(cut=cut):
                args = frozen_run(self.root / str(cut), cut=cut)
                path = Path(args.run) / "frozen_selection.json"
                saved = json.loads(path.read_text())
                saved["metrics"]["track_purity_global"] = saved_purity
                path.write_text(json.dumps(saved))
                self.assertIsNone(self.run_cli(args)["replay_minus_saved_validation"]["track_purity_global"])


class ExatrkxDiagnosticTests(unittest.TestCase):
    def test_missing_or_duplicate_checkpoints_are_rejected(self):
        selected = [{"stage": name} for name in ("embedding", "filter", "gnn")]
        validate_selected_stages(selected)
        for invalid in (selected[:-1], selected + selected[:1], selected[:2] + selected[:1]):
            with self.assertRaises(ValueError):
                validate_selected_stages(invalid)

    def test_other_track_and_background_cannot_bridge_a_true_track(self):
        truth = np.array([0, 0, 1, 1, -1])
        # Both track-1 points reach track-0, but no true track-1 edge exists.
        edges = np.array([[0, 0, 0, 2, 3], [1, 2, 3, 4, 4]])
        row, oracle, tracks = graph_diagnostics(truth, edges)
        self.assertEqual(row["fully_connected_tracks"], 1)
        self.assertEqual(row["true_edges"], 1)
        self.assertEqual(tracks[1]["components"], 2)
        np.testing.assert_array_equal(oracle[2:], [-1, -1, -1])
        self.assertEqual(oracle[0], oracle[1])

    def test_complete_true_edges_yield_truth_partition_and_background(self):
        truth = np.array([0, 0, 1, 1, -1])
        edges = np.array([[0, 2, 0, 3], [1, 3, 2, 4]])
        row, oracle, _ = graph_diagnostics(truth, edges)
        summary = aggregate_graphs([row])
        self.assertEqual(summary["true_pair_recall"], 1.)
        self.assertEqual(summary["fully_connected_track_fraction"], 1.)
        self.assertNotEqual(oracle[0], oracle[2])
        self.assertEqual(oracle[-1], -1)

    def test_pair_recall_and_track_connectivity_are_different(self):
        truth = np.zeros(3, dtype=int)
        edges = np.array([[0, 1], [1, 2]])
        row, _, _ = graph_diagnostics(truth, edges)
        summary = aggregate_graphs([row])
        self.assertAlmostEqual(summary["true_pair_recall"], 2/3)
        self.assertEqual(summary["fully_connected_track_fraction"], 1.)
        distance = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
        self.assertEqual(mst_radius(distance), 1.)
        self.assertEqual(distance.max(), 2.)

    def test_empty_edges_and_singletons_have_explicit_policies(self):
        row, oracle, tracks = graph_diagnostics(np.array([0, 0, 1, -1]), np.empty((2, 0), dtype=int))
        self.assertEqual(row["fully_connected_tracks"], 1)
        self.assertEqual(row["singleton_truth_tracks"], 1)
        np.testing.assert_array_equal(oracle, [-1]*4)
        self.assertEqual(tracks[0]["largest_component_hits"], 1)

    def test_singleton_connectivity_does_not_override_empty_oracle_policy(self):
        row, oracle, _ = graph_diagnostics(np.array([0, 1, -1]), np.empty((2, 0), dtype=int), min_size=1)
        self.assertEqual(row["fully_connected_tracks"], 2)
        np.testing.assert_array_equal(oracle, [-1, -1, -1])

    def test_mst_keeps_zero_distance_edges(self):
        self.assertEqual(mst_radius(np.zeros((4, 4))), 0.)
        self.assertEqual(mst_radius(np.array([[0., 0., 2.], [0., 0., 2.], [2., 2., 0.]])), 2.)

    def test_candidate_union_radius_and_k_scans(self):
        z = torch.tensor([[0.], [1.], [2.5], [6.]])
        edges = undirected_candidates(z, 1, 1.5)
        self.assertEqual({tuple(e) for e in edges.T}, {(0, 1), (1, 2)})
        low = {tuple(e) for e in undirected_candidates(z, 1).T}
        high = {tuple(e) for e in undirected_candidates(z, 2).T}
        self.assertTrue(low <= high)
        complete = undirected_candidates(z, len(z) - 1)
        self.assertEqual(complete.shape[1], 6)
        self.assertEqual(undirected_candidates(z[:1], 1).shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
