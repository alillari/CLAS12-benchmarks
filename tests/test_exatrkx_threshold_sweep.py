import unittest

import numpy as np

from clas12_benchmarks.exatrkx_diagnostics import graph_diagnostics
from clas12_benchmarks.exatrkx_threshold_sweep import (paired_delta_bootstrap, recovery_counts,
                                                      strict_track_states)
from clas12_benchmarks.tracking import components


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
