import unittest

import numpy as np
import torch

from clas12_benchmarks.exatrkx_diagnostics import (aggregate_graphs, graph_diagnostics,
                                                  mst_radius, undirected_candidates)


class ExatrkxDiagnosticTests(unittest.TestCase):
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
