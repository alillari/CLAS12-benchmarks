from dataclasses import replace
import os
import unittest

import torch

from clas12_benchmarks.distance_graphs import DistanceGraphClassifier, DistanceGraphConfig


class DistanceGraphTests(unittest.TestCase):
    def setUp(self):
        precision = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, precision)
        torch.set_float32_matmul_precision("highest")

    def check_ragged_events(self, device, architecture):
        torch.manual_seed(2)
        config = DistanceGraphConfig(architecture, width=16, layers=3, propagate=6,
                                     neighbors=3, aggregators=3, query_chunk=4)
        model = DistanceGraphClassifier(4, config).to(device)
        points = torch.randn(4, 11, 4, device=device)
        points[3, 2:7] = points[3, 1]
        points.requires_grad_(True)
        mask = torch.arange(11, device=device) < torch.tensor([0, 1, 5, 11], device=device)[:, None]
        scores = model(points, mask)
        permutation = torch.randperm(11, device=device)
        torch.testing.assert_close(model(points[:, permutation], mask[:, permutation]),
                                   scores[:, permutation], atol=2e-6, rtol=2e-5)
        for i, size in enumerate((0, 1, 5, 11)):
            isolated = model(points[i:i + 1, :size], mask[i:i + 1, :size])
            torch.testing.assert_close(isolated, scores[i:i + 1, :size], atol=2e-6, rtol=2e-5)
        self.assertTrue(scores[~mask].eq(0).all())
        only_one_event, = torch.autograd.grad(scores[2, :5].sum(), points, retain_graph=True)
        self.assertTrue(only_one_event[[0, 1, 3]].eq(0).all())
        scores[mask].square().mean().backward()
        for layer in model.layers:
            distance = layer.space if architecture == "gravnet" else layer.distance
            for parameter in distance.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def check_high_multiplicity(self, device, architecture):
        torch.manual_seed(3)
        config = DistanceGraphConfig(architecture, width=8, layers=1, propagate=4,
                                     neighbors=5, aggregators=2, query_chunk=37)
        model = DistanceGraphClassifier(4, config).to(device)
        points = torch.randn(1, 557, 4, device=device) * 1e3
        mask = torch.ones(1, 557, dtype=torch.bool, device=device)
        result = model(points, mask)
        self.assertEqual(result.shape, (1, 557, 2))
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        if architecture == "gravnet":
            large_chunk = DistanceGraphClassifier(4, replace(config, query_chunk=900)).to(device)
            large_chunk.load_state_dict(model.state_dict())
            torch.testing.assert_close(large_chunk(points, mask), result, atol=2e-5, rtol=2e-5)

    def test_ragged_events_cpu(self):
        for architecture in ("gravnet", "garnet"):
            with self.subTest(architecture=architecture):
                self.check_ragged_events("cpu", architecture)

    def test_high_multiplicity_cpu(self):
        for architecture in ("gravnet", "garnet"):
            with self.subTest(architecture=architecture):
                self.check_high_multiplicity("cpu", architecture)

    @unittest.skipUnless(os.getenv("TEST_GRAPH_CUDA") == "1", "CUDA graph tests are opt-in")
    def test_distance_graphs_cuda(self):
        for architecture in ("gravnet", "garnet"):
            with self.subTest(architecture=architecture):
                self.check_ragged_events("cuda", architecture)
                self.check_high_multiplicity("cuda", architecture)

    def test_gravnet_distance_ties_across_permutation_and_padding(self):
        torch.manual_seed(7)
        config = DistanceGraphConfig("gravnet", width=8, layers=1, propagate=4, neighbors=2)
        model = DistanceGraphClassifier(4, config)
        with torch.no_grad():
            model.layers[0].space.weight.zero_()
        points = torch.randn(2, 33, 4)
        mask = torch.arange(33) < torch.tensor([9, 33])[:, None]
        scores = model(points, mask)
        permutation = torch.randperm(33)
        torch.testing.assert_close(scores[:, permutation], model(points[:, permutation], mask[:, permutation]))
        torch.testing.assert_close(scores[:1, :9], model(points[:1, :9], mask[:1, :9]))
