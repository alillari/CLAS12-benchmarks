import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class PairedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "paired_comparison", Path(__file__).resolve().parents[1] / "scripts/run_tuned_comparison.py")
        cls.comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.comparison)

    def compare(self, left, right):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("left.jsonl", "right.jsonl")]
            for path, rows in zip(paths, (left, right)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = self.comparison.paired_bootstrap(*paths, replicates=32)
            json.dumps(result, allow_nan=False)
            return result

    def test_undefined_purity_does_not_erase_defined_deltas(self):
        a = dict(identity={"event": 1}, n_matched_tracks=1, n_true_tracks=2, n_pred_tracks=1, ari_signal=.75)
        b = dict(a, n_matched_tracks=0, n_pred_tracks=0, ari_signal=.25)
        result = self.compare([a], [b])
        self.assertEqual(result["delta_95_percent_intervals"], {
            "track_efficiency_global": [.5, .5], "track_purity_global": None, "ari_signal": [.5, .5]})
        self.assertEqual(result["defined_resamples"]["track_purity_global"], 0)

    def test_all_background_metrics_remain_undefined(self):
        row = dict(identity={"event": 1}, n_matched_tracks=0, n_true_tracks=0, n_pred_tracks=0, ari_signal=None)
        result = self.compare([row], [row])
        self.assertTrue(all(value is None for value in result["delta_95_percent_intervals"].values()))

    def test_paired_identity_mismatch_is_rejected(self):
        a = dict(identity={"event": 1}, n_matched_tracks=1, n_true_tracks=1, n_pred_tracks=1, ari_signal=1.)
        with self.assertRaisesRegex(ValueError, "identities differ"):
            self.compare([a], [dict(a, identity={"event": 2})])


if __name__ == "__main__":
    unittest.main()
