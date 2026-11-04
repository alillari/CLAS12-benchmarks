import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from clas12_benchmarks.interaction_network import SegmentClassifier
from clas12_benchmarks.tracking import candidates, filter_probabilities, graph_scores, pair_labels, validate_recipe
from clas12_benchmarks.upstream import ROOT, definition, exatrkx_embedding, exatrkx_filter


class UpstreamFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / "configs/tracking.json").read_text())
        spec = importlib.util.spec_from_file_location("upstream_checks", ROOT / "scripts/check_tracking_upstream.py")
        cls.checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.checks)

    def events(self):
        xyz = np.random.default_rng(4).normal(size=(12, 3)).astype(np.float32)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        positions = np.column_stack((np.arcsinh(xyz[:, 2] / r), xyz[:, 0] / r, xyz[:, 1] / r, r))
        event = SimpleNamespace(raw_features=xyz, positions=positions, labels={"seg_target": np.arange(12)})
        permutation = np.random.default_rng(8).permutation(len(xyz))
        other = SimpleNamespace(raw_features=xyz[permutation], positions=positions[permutation],
                                labels={"seg_target": np.zeros(12, dtype=int)})
        return event, other, permutation

    def test_filter_and_gnn_scores_are_invariant_to_point_order(self):
        torch.manual_seed(7)
        config = self.config["exatrkx"]
        embedding, filt = exatrkx_embedding(config["embedding"]).eval(), exatrkx_filter(config["filter"]).eval()
        gnn = SegmentClassifier(4, config["gnn_processing_steps"]).eval()
        event, other, permutation = self.events()
        left, right = candidates(embedding, event, config, "cpu"), candidates(embedding, other, config, "cpu")
        # Direction, not just the undirected pair, must survive reindexing.
        original = [tuple(edge) for edge in left.T.tolist()]
        mapped = [tuple(permutation[edge].tolist()) for edge in right.T.tolist()]
        lookup = {edge: i for i, edge in enumerate(mapped)}
        self.assertEqual(set(original), set(mapped))
        index = [lookup[edge] for edge in original]
        a = filter_probabilities(filt, event, left, "cpu")
        b = filter_probabilities(filt, other, right, "cpu")
        torch.testing.assert_close(a, b[index], rtol=1e-5, atol=1e-6)
        a, b = graph_scores(gnn, [event, other], [left, right], "cpu")
        torch.testing.assert_close(a, b[index], rtol=1e-5, atol=1e-6)
        radii = np.linalg.norm(event.raw_features, axis=1)
        self.assertTrue(np.all(radii[left[0]] <= radii[left[1]]))

    def test_unnamed_historical_orientation_remains_legacy(self):
        torch.manual_seed(7)
        config = copy.deepcopy(self.config["exatrkx"])
        embedding = exatrkx_embedding(config["embedding"]).eval()
        event, _, _ = self.events()
        config.pop("edge_orientation")
        historical = candidates(embedding, event, config, "cpu")
        explicit = candidates(embedding, event, dict(config, edge_orientation="legacy_index"), "cpu")
        torch.testing.assert_close(historical, explicit, rtol=0, atol=0)
        self.assertTrue(bool((historical[0] < historical[1]).all()))

    def test_new_run_defaults_and_invalid_objectives_are_explicit(self):
        config = copy.deepcopy(self.config)
        config.pop("eggnet_objective")
        config["exatrkx"].pop("edge_orientation")
        validate_recipe(config)
        self.assertEqual(config["eggnet_objective"], "upstream_contrastive")
        self.assertEqual(config["exatrkx"]["edge_orientation"], "geometry")
        for key, value in (("objective", "upstream_constrastive"), ("orientation", "unknown")):
            invalid = copy.deepcopy(config)
            if key == "objective":
                invalid["eggnet_objective"] = value
            else:
                invalid["exatrkx"]["edge_orientation"] = value
            with self.assertRaises(ValueError):
                validate_recipe(invalid)

    def test_reference_kernels_and_independent_forwards(self):
        torch.manual_seed(72)
        self.checks.check_exatrkx_losses("cpu")
        self.checks.check_exatrkx_forwards(self.config["exatrkx"], "cpu")
        self.checks.check_eggnet(self.config["eggnet"], "cpu")
        self.checks.check_components()

    def test_background_and_signal_zero_match_upstream_eggnet_targets(self):
        reference = definition("vendor/eggnet/eggnet/utils/mapping.py", "get_target", {})
        labels = torch.tensor([-1, -1, 0, 0, 1, 1])
        edges = torch.triu_indices(len(labels), len(labels), offset=1)
        torch.testing.assert_close(pair_labels(labels, edges), reference(edges, labels + 1) > 0)


if __name__ == "__main__":
    unittest.main()
