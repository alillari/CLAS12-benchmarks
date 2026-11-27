import unittest

import numpy as np

from clas12_benchmarks.graph_noise_metrics import (
    bootstrap_rates, calibration, event_confusions, operating_points,
)


class GraphNoiseMetricTests(unittest.TestCase):
    def test_calibrate_strict_score_thresholds_including_ties(self):
        labels = np.array([0] * 4 + [1] * 5, dtype=np.int8)
        scores = np.array([.01, .4, .7, .8, .2, .2, .8, .9, .9], dtype=np.float32)
        points = calibration(labels, scores, scores > .5)
        results = operating_points(labels, scores, points)
        self.assertEqual(results["retain_99p5pct"]["signal_retention"], 1.0)
        self.assertEqual(results["retain_99p5pct"]["noise_rejection"], .4)
        self.assertEqual(results["reject_95pct"]["noise_rejection"], 1.0)
        self.assertEqual(results["reject_95pct"]["signal_retention"], .25)


    def test_events_bootstrap_preserves_paired_differences_and_empty_events(self):
        labels = np.array([0, 0, 1, 1, 1])
        prediction = np.array([0, 1, 1, 1, 1], dtype=bool)
        offsets = np.array([0, 3, 3, 5])
        confusion = event_confusions(labels, prediction, offsets)
        np.testing.assert_array_equal(confusion, [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 2]])
        report = bootstrap_rates({"a": confusion, "b": confusion}, draws=40, seed=7)
        paired = report["paired_difference_95pct"]["b_minus_a"]
        self.assertEqual(paired, {"signal_retention": [0.0, 0.0], "noise_rejection": [0.0, 0.0]})
