import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from clas12_benchmarks.interaction_network import SegmentClassifier
from clas12_benchmarks.tracking import (HitBatch, components, edge_loss, embedding_loss,
                                        pair_labels, train_stage, eggnet_contrastive_loss, learning_rate_at_step)
from clas12_benchmarks.upstream import (ROOT, eggnet, exatrkx_embedding, exatrkx_filter,
                                       neighbor_edges, scatter_add, softmax)
from clas12_benchmarks.prepare_cohort import prepare
from clas12_benchmarks.upstream import definition


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / "configs/first_real_runs.json").read_text())

    def test_segment_reductions_values_and_gradients(self):
        x = torch.tensor([[100., -2.], [-100., 3.], [1., 5.], [2., 4.]], requires_grad=True)
        index = torch.tensor([2, 0, 2, 0])
        result = softmax(x, index)
        expected = torch.empty_like(x)
        for i in (0, 2):
            expected[index == i] = x[index == i].softmax(0)
        torch.testing.assert_close(result, expected)
        (result * torch.arange(8).reshape(4, 2)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.testing.assert_close(scatter_add(x, index, dim_size=4)[1], torch.zeros(2))

    def test_knn_duplicate_positions_no_self_and_radius(self):
        x = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [3., 0.]])
        edges = neighbor_edges(x, 10, radius=1.)
        self.assertEqual(set(map(tuple, edges.T.tolist())), {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)})
        self.assertEqual(neighbor_edges(x[:1], 10).shape, (2, 0))

    def test_background_pairs_never_form_positive_edges(self):
        labels = torch.tensor([-1, -1, 0, 0, 1])
        edges = torch.tensor([[0, 2, 3], [1, 3, 4]])
        self.assertEqual(pair_labels(labels, edges).tolist(), [False, True, False])
        z = torch.randn(5, 3, requires_grad=True)
        loss = embedding_loss(z, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        singleton = z[:1]
        self.assertEqual(float(embedding_loss(singleton, labels[:1])), 0.)
        with self.assertRaisesRegex(ValueError, "limit"):
            embedding_loss(z, labels, max_pairs=2)

    def test_corrected_eggnet_loss_matches_actual_upstream_values_and_gradients(self):
        get_distances = definition("vendor/eggnet/eggnet/loss/utils/utils.py", "get_distances", {})
        upstream_loss = definition("vendor/eggnet/eggnet/loss/utils/utils.py", "hinge_loss", {"get_distances": get_distances})
        z = torch.tensor([[0., .1], [.2, .3], [.6, -.1], [.1, .2]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, -1])
        torch.manual_seed(23)
        actual = eggnet_contrastive_loss(z, labels, k=2, random_pairs_per_node=5)
        actual_gradient = torch.autograd.grad(actual, z, retain_graph=True)[0]
        pairs = torch.triu_indices(4, 4, offset=1)
        signal = pairs[:, pair_labels(labels, pairs)]
        hard = neighbor_edges(z, 2)
        torch.manual_seed(23)
        random = torch.randint(4, (2, 20))
        random = random[:, random[0] != random[1]]
        batch = SimpleNamespace(hit_embedding=z)
        expected = sum(upstream_loss(batch, edges, 1., y=pair_labels(labels, edges).float() * 2 - 1,
                                    w=torch.ones(edges.shape[1])) for edges in (signal, hard, random))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, z)[0])

    def test_corrected_repulsion_remains_strong_near_collapse(self):
        z = torch.tensor([[0.], [.0001]], requires_grad=True)
        labels = torch.tensor([0, 1])
        torch.manual_seed(4)
        loss = eggnet_contrastive_loss(z, labels)
        grad = torch.autograd.grad(loss, z)[0]
        self.assertGreater(float(grad.abs().max()), 1.)
        self.assertGreater(float(grad[0]), 0.)

    def test_warmup_cosine_endpoints(self):
        c = dict(self.config, learning_rate_schedule="warmup_cosine", warmup_steps=10,
                 min_learning_rate=.00001, steps_per_stage=100)
        self.assertAlmostEqual(learning_rate_at_step(c, 0), c["learning_rate"] / 10)
        self.assertAlmostEqual(learning_rate_at_step(c, 9), c["learning_rate"])
        self.assertAlmostEqual(learning_rate_at_step(c, 99), c["min_learning_rate"])

    def test_architectures_update_and_strict_reload(self):
        torch.manual_seed(11)
        x = torch.randn(14, 12)
        graph = neighbor_edges(x[:, :4], 4)
        models = [eggnet(self.config["eggnet"]), exatrkx_embedding(self.config["exatrkx"]["embedding"]),
                  exatrkx_filter(self.config["exatrkx"]["filter"]), SegmentClassifier(num_processing_steps=3)]
        def forward(i, model):
            if i == 0:
                return model(HitBatch(**{f"hit_f{j}": x[:, j] for j in range(12)}))
            if i == 1:
                return model(x[:, :4])
            if i == 2:
                return model(x[:, :4], graph)
            return torch.cat(model(x[:, :4], graph))
        with tempfile.TemporaryDirectory() as directory:
            for i, model in enumerate(models):
                before = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                forward(i, model).square().mean().backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
                path = Path(directory) / "checkpoint.pt"
                torch.save(model.state_dict(), path)
                expected = forward(i, model).detach()
                model.load_state_dict(torch.load(path, weights_only=True), strict=True)
                torch.testing.assert_close(forward(i, model), expected, rtol=0, atol=0)

    def test_interaction_network_against_numpy_graph_blocks(self):
        # Independent dense incidence matrix formulation checks source/destination
        # ordering, both aggregates, node-before-edge updates and latent skips.
        torch.manual_seed(6)
        model = SegmentClassifier(input_dim=4, num_processing_steps=3).double()
        x = np.random.default_rng(4).normal(size=(5, 4))
        graph = np.array([[0, 2, 4, 3], [1, 1, 0, 2]])
        sender, receiver = graph
        def dense_mlp(layers, value):
            for layer in layers:
                if isinstance(layer, torch.nn.Linear):
                    value = value @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
                else:
                    value = np.maximum(value, 0)
            return value
        n0 = dense_mlp(model.node_encoder, x)
        e0 = dense_mlp(model.edge_encoder, np.c_[n0[receiver], n0[sender]])
        nodes, edges = n0, e0
        incoming, outgoing = np.zeros((5, 4)), np.zeros((5, 4))
        incoming[receiver, np.arange(4)] = 1
        outgoing[sender, np.arange(4)] = 1
        expected = []
        for _ in range(3):
            node_input, edge_input = np.c_[n0, nodes], np.c_[e0, edges]
            nodes = dense_mlp(model.node_core, np.c_[incoming @ edge_input, outgoing @ edge_input, node_input])
            edges = dense_mlp(model.edge_core, np.c_[edge_input, nodes[receiver], nodes[sender]])
            expected.append(dense_mlp([model.edge_output], edges).ravel())
        actual = model(torch.tensor(x), torch.tensor(graph))
        np.testing.assert_allclose(torch.stack(actual).detach().numpy(), np.array(expected), rtol=1e-10, atol=1e-12)

    def test_components_keep_rejected_nodes_as_background(self):
        labels = components(7, [[0, 1, 3], [1, 2, 4]], min_size=2)
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[3])
        np.testing.assert_array_equal(labels[5:], [-1, -1])
        np.testing.assert_array_equal(components(3, np.empty((2, 0))), [-1, -1, -1])

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        # Covers optimizer, step position and shuffle RNG over an epoch boundary.
        events = [torch.tensor([[i, 1.]], dtype=torch.float32) for i in range(1, 4)]
        config = dict(self.config, steps_per_stage=5, validation_interval=1)
        def objective(m, event):
            return m(event).square().mean()
        torch.manual_seed(12)
        a = torch.nn.Linear(2, 1)
        b = torch.nn.Linear(2, 1)
        b.load_state_dict(a.state_dict())
        with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
            train_stage(a, "test", events, events, objective, config, Path(da), torch.device("cpu"))
            train_stage(b, "test", events, events, objective, dict(config, steps_per_stage=2), Path(db), torch.device("cpu"))
            train_stage(b, "test", events, events, objective, config, Path(db), torch.device("cpu"), resume=True)
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)

    def test_scheduled_checkpoint_resume_and_score_selection(self):
        events = [torch.tensor([[i, 1.]], dtype=torch.float32) for i in range(1, 4)]
        config = dict(self.config, steps_per_stage=5, validation_interval=1,
                      learning_rate_schedule="warmup_cosine", warmup_steps=2, min_learning_rate=.00001)
        def objective(m, event):
            return m(event).square().mean()
        torch.manual_seed(12)
        a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        b.load_state_dict(a.state_dict())
        with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
            train_stage(a, "test", events, events, objective, config, Path(da), torch.device("cpu"))
            train_stage(b, "test", events, events, objective, dict(config, steps_per_stage=2), Path(db), torch.device("cpu"))
            train_stage(b, "test", events, events, objective, config, Path(db), torch.device("cpu"), resume=True)
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            selected = train_stage(a, "score", events, events, objective, config, Path(directory), torch.device("cpu"),
                score_fn=lambda model, data: float(objective(model, data[0])))
            # Deliberately maximize this monotonically decreasing objective to
            # check that the callback, rather than minimum loss, selects step 1.
            self.assertEqual(selected["selected_step"], 1)

    def test_real_cohort_preparation_contract_on_native_fixture(self):
        from mmap_ninja import RaggedMmap
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "metadata.json").write_text("{}")
            for split in ("pretrain", "test"):
                xyz = [np.array([[1, 0, 1], [0, 2, 2], [3, 0, 3]], np.float32) * (i + 1) for i in range(10)]
                ys = [np.array([0, 0, -1], dtype=np.int64) for _ in xyz]
                for name, values in (("features", xyz), ("seg_target", ys), ("coatjava_seg_pred", ys)):
                    RaggedMmap.from_generator(root / f"{name}_{split}", iter(values), batch_size=10)
                (root / f"samples_{split}.jsonl").write_text("".join(json.dumps({"source_file": split + ".hipo", "event": i,
                    "sample_mode": "event"}) + "\n" for i in range(10)))
            output = Path(directory) / "subset"
            cohort = prepare(root, output, train=4, val=2, test=3, seed=3)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["roles"]["train"]["n_events"], 4)
            self.assertEqual(len(cohort.roles["test"]), 3)
            self.assertEqual(cohort.event("val", 0).labels["coatjava_seg_pred"].tolist(), [0, 0, -1])
            expanded = prepare(root, Path(directory) / "expanded", train=4, val=3, test=5, seed=7, extend_from=output)
            def identities(c, role):
                return {(c.event(role, i).identity["source_file"], c.event(role, i).identity["event"])
                        for i in range(len(c.roles[role]))}
            self.assertEqual(identities(cohort, "train"), identities(expanded, "train"))
            self.assertTrue(identities(cohort, "val") <= identities(expanded, "val"))
            self.assertFalse(identities(cohort, "test") & identities(expanded, "test"))
            self.assertEqual(next(iter(cohort.datasets.values()))[0]["radius_scale"],
                             next(iter(expanded.datasets.values()))[0]["radius_scale"])


if __name__ == "__main__":
    unittest.main()
