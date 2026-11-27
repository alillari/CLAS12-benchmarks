"""Cover measured geometry, train/validation isolation and keyed momentum scoring."""

import csv
import gzip
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from clas12_benchmarks.cluster_mlp import decode
from clas12_benchmarks.cluster_mlp_features import ARMS, encode, rotate
from clas12_benchmarks.cluster_momentum_geometry import aggregate_layers
from clas12_benchmarks.hdf5_momentum import build_manifest
from clas12_benchmarks.momentum_evaluation import evaluate
from clas12_benchmarks.momentum_manifest import TrackManifest
from clas12_benchmarks.prepare_cohort import prepare
from hdf5_fixture import make_publication


class ClusterMLPTests(unittest.TestCase):
    def test_aggregation_and_rotation(self):
        xyz = np.array([[3, 6, 9], [9, 0, 3], [12, 3, 6], [21, 3, 6], [24, 0, 3]], dtype=np.float32)
        endpoints = np.column_stack((xyz + [0, 1, -2], xyz + [0, -1, 2])).astype(np.float32)
        geometry = aggregate_layers(xyz, endpoints, np.array([2, 2, 2, 1, 1]), np.array([1, 1, 2, 2, 5]))
        np.testing.assert_allclose(geometry[0, :3], [0.2, 0.1, 0.2])
        np.testing.assert_allclose(geometry[0, 9], np.log1p(2))
        np.testing.assert_array_equal(geometry[2:6], 0.)
        self.assertEqual(np.count_nonzero(geometry[:, 9]), 4)
        raw = torch.from_numpy(geometry[None])
        rotation = torch.tensor([[0., 1.]])
        rotated = raw.clone()
        rotated[..., :9] = rotate(raw[..., :9].reshape(1, 12, 3, 3), rotation, inverse=True).flatten(2)
        stats = {"y_mean": torch.zeros(3), "y_std": torch.ones(3)}
        prediction = torch.tensor([[.1, .2, -.3]])
        for arm, columns in ARMS.items():
            with self.subTest(arm=arm):
                features, frame = encode(raw, arm)
                features_rotated, frame_rotated = encode(rotated, arm)
                self.assertEqual(features.shape, (1, len(columns)))
                torch.testing.assert_close(features_rotated, features, atol=1e-5, rtol=0)
                expected = rotate(decode(prediction, frame, stats), rotation, inverse=True)
                torch.testing.assert_close(decode(prediction, frame_rotated, stats), expected)

    def test_native_pack_fit_and_gzip_evaluation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = np.array([[9, 0, 1], [10, 1, 1], [12, 2, 2], [18, 3, 3],
                            [22, 5, 4], [8, 5, 2]], dtype=np.float32)
            owners = np.array([17, 17, 17, 17, 17, 42], dtype=np.int32)
            fields = {
                "xyz": xyz, "endpoints": np.column_stack((xyz - 1, xyz + 1)),
                "detector": np.array([2, 2, 2, 1, 1, 2], dtype=np.int32),
                "layer": np.array([1, 1, 2, 2, 4, 1], dtype=np.int32),
                "primary_segment": np.where(owners == 17, 3, 9).astype(np.int32),
                "reco_segment": np.where(owners == 17, 0, 1).astype(np.int32),
                "primary_tid": owners, "primary_pid": np.full(6, 211, np.int32),
                "primary_pid_valid": np.ones(6, bool), "primary_noise": np.zeros(6, np.int8),
                "primary_noise_valid": np.ones(6, bool),
            }
            raw = make_publication(root / "raw", cluster_fields=fields)
            with prepare(raw, root / "cohort", reader="hdf5", train=3, val=2, test=1, seed=42):
                pass
            path = root / "tracks.npz"
            build_manifest(root / "cohort/cohort.json", path, min_clusters=1, cluster_mlp_geometry=True)
            tracks = TrackManifest(path)
            self.assertEqual(tracks.meta["selection_counts"]["evaluation"]["layer_coverage_excluded"], 1)
            self.assertEqual(tracks.identity(tracks.rows("evaluation")[0])[-1], 17)

            weights = root / "model.pt"
            predictions = root / "predictions.csv.gz"
            cli = [sys.executable, "-m", "clas12_benchmarks.cluster_mlp"]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            subprocess.run([*cli, "train", "--manifest", str(path), "--output", str(weights),
                            "--steps", "2", "--batch", "2", "--device", "cpu"],
                           check=True, capture_output=True, env=env)
            checkpoint = torch.load(weights, weights_only=True)
            self.assertEqual(checkpoint["training_source"]["tracks"], 3)
            self.assertEqual(checkpoint["validation_source"]["tracks"], 2)
            self.assertEqual(checkpoint["step"], 2)
            subprocess.run([*cli, "predict", "--manifest", str(path), "--checkpoint", str(weights),
                            "--output", str(predictions), "--device", "cpu"],
                           check=True, capture_output=True, env=env)
            with gzip.open(predictions, "rt") as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(row["manifest_sha256"], tracks.sha256)
            self.assertEqual(evaluate(tracks, predictions)["all_tracks"]["n"], 1)
