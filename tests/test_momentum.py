import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from mmap_ninja import RaggedMmap

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.momentum_evaluation import evaluate, verify_score_tables
from clas12_benchmarks.momentum_manifest import TrackManifest, build_manifest


class MomentumTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.raw = self.dir / "raw"
        self.raw.mkdir()
        meta = {"feature_units": "cm", "feature_layout": ["x", "y", "z"],
                "seg_target_source": "mctrue_primary_tid",
                "reg_target_layout": ["mc_entrance_pz", "unused", "mc_entrance_px", "mc_entrance_py"],
                "aux_target_layout": ["cvttraj_entrance_phi", "cvt_benchmark_p", "cvt_benchmark_theta"]}
        (self.raw / "metadata.json").write_text(json.dumps(meta))
        seg = np.r_[[-1], np.tile([4, 2], 12)].astype(np.int64)
        xyz = np.arange(len(seg)*3, dtype=np.float32).reshape(-1, 3)
        truth = np.full((len(seg), 4), np.nan, np.float32)
        truth[seg == 2] = [3000, np.nan, 1000, -2000]
        truth[seg == 4] = [0, np.nan, -1000, 0]
        truth[2, 0] = np.nan  # independently finite component reduction
        aux = np.full((len(seg), 3), np.nan, np.float32)
        aux[seg == 2] = [np.pi/2, 3, np.pi/2]
        aux[2, 0] = np.nan  # skip the first incomplete auxiliary row
        arrays = (xyz, seg, truth, aux)
        self.raw_arrays = arrays
        for split, events in (("test", [46, 91]), ("pretrain", [92])):
            (self.raw / f"samples_{split}.jsonl").write_text("".join(json.dumps(
                {"source_file": "source.hipo", "event": e, "sample_mode": "event"})+"\n" for e in events))
            for field, array in zip(("features", "seg_target", "reg_target", "aux_target"), arrays):
                RaggedMmap.from_generator(self.raw / f"{field}_{split}",
                                          (array.copy() for _ in events), batch_size=len(events))
        h = hashlib.sha256()
        for a in arrays[:3]:
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        self.digest = h.hexdigest()
        self.infos = {}
        for role, items in (("train_1000", [(0, 4)]), ("train_50000", [(0, 2), (0, 4)]),
                            ("validation", [(0, 4)]), ("evaluation", [(1, 4), (1, 2)])):
            self.keys_file(role, items)
        self.write_ready()
        self.output = self.dir / "tracks.npz"

    def keys_file(self, role, items):
        keys = np.asarray(items, dtype=np.int64)
        lengths = np.full(len(items), 12, np.int64)
        np.savez(self.dir / f"momentum_{role}.npz", keys=keys, lengths=lengths)
        self.infos[role] = {"count": len(keys), "keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
                            "lengths_sha256": hashlib.sha256(lengths.tobytes()).hexdigest(),
                            "source_content_sha256": self.digest}

    def write_ready(self):
        self.infos["train"] = self.infos["train_50000"].copy()
        ready = {"complete": True, "metadata_sha256": file_sha256(self.raw / "metadata.json"),
                 "cohorts": {"momentum": self.infos}}
        (self.dir / "data_ready.json").write_text(json.dumps(ready))

    def manifest(self):
        build_manifest(self.raw, self.dir, self.output)
        return TrackManifest(self.output)

    def test_segment_truth_reference_original_indices_and_budgets(self):
        m = self.manifest()
        self.assertEqual(m.rows("train_1000").tolist(), [1])
        rows = m.rows("evaluation")
        self.assertEqual([m.identity(i) for i in rows], [("source.hipo", 91, 4), ("source.hipo", 91, 2)])
        np.testing.assert_array_equal(m.truth[rows], [[-1, 0, 0], [1, -2, 3]])
        self.assertTrue(np.isnan(m.cvt[rows[0]]).all())
        np.testing.assert_allclose(m.cvt[rows[1]], [0, 3, 0], atol=2e-7)
        hits, indices = m.hits(rows[1])
        np.testing.assert_array_equal(indices, np.arange(2, 25, 2))
        np.testing.assert_array_equal(hits, self.raw_arrays[0][indices])
        with self.assertRaises(FileExistsError):
            self.manifest()

    def test_raw_drift_and_event_leakage_fail_before_publish(self):
        self.infos["evaluation"]["source_content_sha256"] = "0"*64
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.manifest()
        self.assertFalse(self.output.exists())
        self.infos["evaluation"]["source_content_sha256"] = self.digest
        self.write_ready()
        path = self.raw / "samples_pretrain.jsonl"
        path.write_text(path.read_text().replace("92", "91"))
        with self.assertRaisesRegex(ValueError, "Repeated source event"):
            self.manifest()
        self.assertFalse(self.output.exists())

    def test_non_nested_track_or_invalid_feature_order_rejected(self):
        self.keys_file("train_1000", [(0, 8)])
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.manifest()
        path = self.raw / "metadata.json"
        meta = json.loads(path.read_text())
        meta["feature_layout"] = ["y", "x", "z"]
        path.write_text(json.dumps(meta))
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "cluster centers"):
            self.manifest()

    def test_target_change_cannot_change_features(self):
        before = self.manifest()
        self.output = self.dir / "different.npz"
        xyz, seg, reg, aux = self.raw_arrays
        reg = reg * 2
        aux = aux * 0.5
        for split, count in (("pretrain", 1), ("test", 2)):
            for field, changed in (("reg_target", reg), ("aux_target", aux)):
                array = RaggedMmap(str(self.raw / f"{field}_{split}"), mode="r+")
                for i in range(count):
                    array[i][:] = changed
        h = hashlib.sha256()
        for a in (xyz, seg, reg):
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        for entry in self.infos.values():
            entry["source_content_sha256"] = h.hexdigest()
        self.write_ready()
        after = self.manifest()
        np.testing.assert_array_equal(before.xyz, after.xyz)
        np.testing.assert_array_equal(before.point_index, after.point_index)
        np.testing.assert_array_equal(before.truth * 2, after.truth)

    def test_keyed_vector_pt_and_integrity_checks(self):
        m = self.manifest()
        rows = m.rows("evaluation")
        path = self.dir / "predictions.csv"
        def write(items, scalar=False):
            fields = ["source_file", "event", "segment_label"]
            fields += ["pred_pt_gev"] if scalar else [f"pred_{axis}_gev" for axis in ("px", "py", "pz")]
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                for i, values in items:
                    writer.writerow([*m.identity(i), *values])
        perfect = [(i, m.truth[i]) for i in rows]
        write(perfect[::-1])
        first = evaluate(m, path)
        self.assertEqual(first["all_tracks"]["p_mae_mev"], 0)
        self.assertEqual(first["cvt_valid"]["model"]["n"], 1)
        self.assertEqual(first["n_missing_cvt"], 1)
        write(perfect)
        self.assertEqual(first["all_tracks"], evaluate(m, path)["all_tracks"])
        write([(i, [np.hypot(*m.truth[i, :2])]) for i in rows], scalar=True)
        self.assertEqual(evaluate(m, path)["all_tracks"]["pt_mae_mev"], 0)
        for items, message in ((perfect[:1], "Missing"), (perfect + perfect[:1], "duplicate"),
                               ([(rows[0], [0, 0, np.nan]), perfect[1]], "nonfinite")):
            write(items)
            with self.assertRaisesRegex(ValueError, message):
                evaluate(m, path)
        for fname in ("metrics_all_50000_tracks.csv", "metrics_common_population.csv"):
            (self.dir / fname).write_text("model,n\nwrong,2\n")
        with self.assertRaisesRegex(ValueError, "Expected one reference"):
            verify_score_tables(first, self.dir, "fixture")

    def test_companion_rows_must_match_independent_raw_vectors_and_membership(self):
        m = self.manifest()
        rows = []
        for i in reversed(m.rows("evaluation")):
            source, event, label = m.identity(i)
            row = dict(source_file=source, event=event, segment_label=label, truth_segment_label=label,
                       source_event_index=1, n_hits=12)
            for prefix, vector in (("true", m.truth[i]), ("adapter", m.truth[i]), ("cvt_entrance", m.cvt[i])):
                row.update({f"{prefix}_{axis}_gev": value for axis, value in zip(("px", "py", "pz"), vector)})
            rows.append(row)
        path = self.dir / "companion.csv"
        def score():
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return evaluate(m, path, companion=True)
        self.assertEqual(score()["companion_row_audit"]["truth_max_abs_difference_gev"], 0)
        for field, value, error in (("source_event_index", 0, "membership"), ("n_hits", 20, "membership"),
                                    ("true_px_gev", 1000, "Archived truth"), ("cvt_entrance_px_gev", 1, "Archived cvt"),
                                    ("source_file", "other.hipo", "Unexpected")):
            old, rows[0][field] = rows[0][field], value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                score()
            rows[0][field] = old
