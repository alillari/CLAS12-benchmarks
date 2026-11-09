"""Compare the Torch Interaction Network with the unmodified TF/Sonnet source.

Optional verification environment: tensorflow-cpu==2.20.0, dm-sonnet==2.0.2,
graph-nets==1.1.0, plus the tracking extra. Not needed for training.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from graph_nets import graphs
import torch

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.interaction_network import SegmentClassifier
from clas12_benchmarks.tracking import edge_loss
from clas12_benchmarks.upstream import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    torch.set_num_threads(2)
    source = ROOT / "vendor/exatrkx/exatrkx/src/tfgraphs/model.py"
    spec = importlib.util.spec_from_file_location("exatrkx_tf_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Execute the actual pinned training-loss function with class weights chosen
    # to express our declared class-mean reduction. The upstream default weights
    # and sampling are separate recipe choices, not numerically equivalent.
    training_source = ROOT / "vendor/exatrkx/exatrkx/scripts/train_gnn_tf.py"
    tree = ast.parse(training_source.read_text())
    loss_functions = [node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "create_loss_ops"]
    if len(loss_functions) != 1:
        raise ValueError("Expected one pinned upstream GNN loss function")
    results = []
    for rounds in (1, 3, 6, 8):
        torch.manual_seed(17)
        rng = np.random.default_rng(23)
        x = rng.normal(size=(11, 4)).astype(np.float32)
        edges = rng.integers(0, 11, size=(2, 31), dtype=np.int32)
        edges = edges[:, edges[0] != edges[1]]
        truth = rng.integers(0, 2, size=edges.shape[1]).astype(np.float32)
        positive_weight = 1.25
        loss_scope = {"tf": tf,
            "real_weight": len(truth) * positive_weight / (truth.sum() * (positive_weight + 1)),
            "fake_weight": len(truth) / ((len(truth) - truth.sum()) * (positive_weight + 1))}
        exec(compile(ast.Module(body=loss_functions, type_ignores=[]), str(training_source), "exec"), loss_scope)
        inp = tf.Variable(x)
        graph = graphs.GraphsTuple(nodes=inp, edges=tf.zeros((edges.shape[1], 1)), globals=tf.zeros((1, 1)),
            senders=tf.constant(edges[0]), receivers=tf.constant(edges[1]), n_node=tf.constant([len(x)]),
            n_edge=tf.constant([edges.shape[1]]))
        ref = reference.SegmentClassifier()
        ref(graph, rounds)
        model = SegmentClassifier(num_processing_steps=rounds)
        variables = list(ref.variables)
        bindings = []
        for layers, scope in ((model.node_encoder, "node_encoder_block/mlp"),
            (model.edge_encoder, "edge_encoder_block/mlp"), (model.node_core, "interaction_network/node_block/mlp"),
            (model.edge_core, "interaction_network/edge_block/mlp"), ([model.edge_output], "graph_independent/edge_model/edge_output")):
            linears = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
            for i, layer in enumerate(linears):
                for suffix, parameter, transpose in (("w", layer.weight, True), ("b", layer.bias, False)):
                    matches = [var for var in variables if var.name.endswith(f"/{scope}/linear_{i}/{suffix}:0")]
                    if len(matches) != 1:
                        raise ValueError(f"Ambiguous reference variable: {scope}/{i}/{suffix}")
                    var = matches[0]
                    var.assign(parameter.detach().numpy().T if transpose else parameter.detach().numpy())
                    bindings.append((var, parameter, transpose))
        with tf.GradientTape() as tape:
            tf_graphs = ref(graph.replace(nodes=inp), rounds)
            tf_out = tf.stack([g.edges[:, 0] for g in tf_graphs])
            targets = type("Targets", (), {"edges": tf.constant(truth)})()
            tf_loss = tf.reduce_mean(loss_scope["create_loss_ops"](targets, tf_graphs))
        tf_gradients = tape.gradient(tf_loss, [inp] + [b[0] for b in bindings])
        torch_input = torch.tensor(x, requires_grad=True)
        logits = model(torch_input, torch.tensor(edges, dtype=torch.long))
        torch_out = torch.stack([output.sigmoid() for output in logits])
        torch_loss = torch.stack([edge_loss(output, torch.tensor(truth, dtype=torch.bool), positive_weight)
                                  for output in logits]).mean()
        np.testing.assert_allclose(float(torch_loss.detach()), float(tf_loss.numpy()), rtol=2e-5, atol=2e-6)
        torch_loss.backward()
        np.testing.assert_allclose(torch_out.detach().numpy(), tf_out.numpy(), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(torch_input.grad.numpy(), tf_gradients[0].numpy(), rtol=2e-4, atol=2e-6)
        gradient_error = float(np.max(np.abs(torch_input.grad.numpy() - tf_gradients[0].numpy())))
        for (_, parameter, transpose), gradient in zip(bindings, tf_gradients[1:], strict=True):
            expected = gradient.numpy().T if transpose else gradient.numpy()
            np.testing.assert_allclose(parameter.grad.numpy(), expected, rtol=2e-4, atol=2e-6)
            gradient_error = max(gradient_error, float(np.max(np.abs(parameter.grad.numpy() - expected))))
        results.append({"processing_steps": rounds, "max_probability_error": float(np.max(np.abs(torch_out.detach().numpy() - tf_out.numpy()))),
                        "max_gradient_error": gradient_error,
                        "training_loss_error": abs(float(torch_loss.detach()) - float(tf_loss.numpy()))})
    report = {"passed": True, "reference_sha256": file_sha256(source),
              "training_source_sha256": file_sha256(training_source),
              "loss_policy": "upstream log_loss with class-mean-equivalent weights; Torch stable BCE logits; mean over rounds",
              "torch": torch.__version__,
              "tensorflow": tf.__version__, "cases": results}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
