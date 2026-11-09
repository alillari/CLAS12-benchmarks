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

from clas12_benchmarks.exatrkx_diagnostics import graph_diagnostics
from clas12_benchmarks.exatrkx_threshold_sweep import (paired_delta_bootstrap, recovery_counts,
                                                      strict_track_states)
from clas12_benchmarks.tracking import components
from clas12_benchmarks import exatrkx_threshold_sweep
from exatrkx_fixture import frozen_run


class SweepIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def run_cli(self, args):
        argv = ["sweep", "--run", args.run, "--cohort", args.cohort, "--output", args.output,
                "--device", "cpu", "--threads", "2", "--cuts", "0", "1", "--bootstrap", "8"]
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            exatrkx_threshold_sweep.main()
        return json.loads((Path(args.output) / "summary.json").read_text())

    def test_cli_accepts_all_storage_fingerprints_and_preserves_scores(self):
        for backend in ("materialized", "external", "hdf5"):
            with self.subTest(backend=backend):
                args = frozen_run(self.root / backend, backend)
                report = self.run_cli(args)
                saved = json.loads((Path(args.run) / "frozen_selection.json").read_text())
                self.assertEqual(report["baseline"]["metrics"], saved["metrics"])
                self.assertEqual(report["events"], 1)
                curve = json.loads((Path(args.output) / "threshold_curve.json").read_text())
                self.assertEqual([row["cut"] for row in curve], [0., .3, 1.])
                self.assertEqual(curve[0]["graph"]["edges"], 10)
                self.assertEqual(curve[-1]["graph"]["edges"], 0)
                self.assertIsNone(curve[-1]["metrics"]["track_purity_global"])
                with np.load(Path(args.output) / "edge_scores.npz") as cache:
                    np.testing.assert_array_equal(cache["scores"], np.full(10, .5))
                    np.testing.assert_array_equal(cache["event_offsets"], [0, 10])
                    np.testing.assert_array_equal(cache["event_point_counts"], [5])

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

    def test_cli_checks_shared_inference_and_reader_sources_before_loading(self):
        for backend, source in (("materialized", "tracking.py"), ("hdf5", "hdf5.py")):
            with self.subTest(backend=backend):
                args = frozen_run(self.root / backend, backend)
                path = Path(args.run) / "provenance.json"
                provenance = json.loads(path.read_text())
                provenance["source_sha256"]["src/clas12_benchmarks/" + source] = "0" * 64
                path.write_text(json.dumps(provenance))
                with patch.object(exatrkx_threshold_sweep, "exatrkx_embedding", side_effect=AssertionError("Must reject before model loading")):
                    with self.assertRaisesRegex(ValueError, "source changed or missing.*" + source):
                        self.run_cli(args)
                self.assertFalse((Path(args.output) / "provenance.json").exists())

    def test_no_predictions_at_saved_cut_still_produces_summary_and_plot(self):
        args = frozen_run(self.root / "empty", cut=1.)
        report = self.run_cli(args)
        self.assertEqual(report["baseline"]["metrics"]["n_pred_tracks"], 0)
        self.assertIsNone(report["baseline"]["metrics"]["track_purity_global"])
        script = Path(__file__).resolve().parents[1] / "scripts/plot_exatrkx_threshold_sweep.py"
        subprocess.run([sys.executable, str(script), "--input", args.output], check=True, capture_output=True, text=True)
        for suffix in ("png", "svg"):
            self.assertGreater((Path(args.output) / f"gnn_threshold_sweep.{suffix}").stat().st_size, 0)


class ThresholdSweepTests(unittest.TestCase):
    def state(self, truth, edges):
        _, _, rows = graph_diagnostics(truth, edges)
        prediction = components(len(truth), edges, min_size=2)
        return strict_track_states(truth, prediction, rows)

    def test_lower_cut_can_recover_connectivity_and_merge_tracks(self):
        truth = np.array([0, 0, 1, 1])
        high = np.array([[0], [1]])
        low = np.array([[0, 2, 1], [1, 3, 2]])
        baseline, candidate = self.state(truth, high), self.state(truth, low)
        np.testing.assert_array_equal(baseline[0], [True, False])
        np.testing.assert_array_equal(candidate[0], [True, True])
        np.testing.assert_array_equal(candidate[1], [False, False])
        counts = recovery_counts(baseline, candidate)
        self.assertEqual(counts["newly_fully_connected_tracks"], 1)
        self.assertEqual(counts["newly_fully_connected_unmerged_tracks"], 0)
        self.assertEqual(counts["newly_fully_connected_merged_tracks"], 1)
        self.assertEqual(counts["baseline_clean_tracks_now_merge_exposed"], 1)

    def test_recovery_without_a_false_bridge_is_clean(self):
        truth = np.array([0, 0, 1, 1])
        high, low = np.array([[0], [1]]), np.array([[0, 2], [1, 3]])
        counts = recovery_counts(self.state(truth, high), self.state(truth, low))
        self.assertEqual(counts["newly_fully_connected_unmerged_tracks"], 1)
        self.assertEqual(counts["baseline_clean_tracks_now_merge_exposed"], 0)

    def test_background_is_not_a_second_signal_track(self):
        truth = np.array([0, 0, -1])
        state = self.state(truth, np.array([[0, 1], [1, 2]]))
        np.testing.assert_array_equal(state[1], [True])
        np.testing.assert_array_equal(state[2], [False])
        self.assertEqual(state[3], 0)

    def test_undefined_purity_keeps_defined_paired_statistics(self):
        def row(matched, predicted):
            return {"metrics": {"n_matched_tracks": matched, "n_true_tracks": 2, "n_pred_tracks": predicted, "ari_signal": .5},
                    "graph": {"fully_connected_tracks": 1}, "clean_fully_connected_tracks": 1}
        result = paired_delta_bootstrap([row(1, 1)], [row(0, 0)], replicates=16)
        self.assertEqual(result["delta_95_intervals"]["track_f1_global"], [2/3, 2/3])
        self.assertIsNone(result["delta_95_intervals"]["track_purity_global"])

    def test_identical_arms_have_zero_paired_deltas(self):
        row = {"metrics": {"n_matched_tracks": 1, "n_true_tracks": 2, "n_pred_tracks": 2, "ari_signal": .5},
               "graph": {"fully_connected_tracks": 1}, "clean_fully_connected_tracks": 1}
        result = paired_delta_bootstrap([row], [row], replicates=16)
        self.assertTrue(all(value == [0., 0.] for value in result["delta_95_intervals"].values()))


if __name__ == "__main__":
    unittest.main()
