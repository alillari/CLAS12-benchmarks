import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.predictions import save_predictions


class TrackingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location(
            "tracking_audit_test", Path(__file__).resolve().parents[1] / "scripts/audit_tracking_predictions.py")
        self.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.audit)
        identity = {"source_file": "fixture.hipo", "event": 7}
        truth = np.array([0, 0, -1])
        event = SimpleNamespace(identity=identity, positions=np.zeros((3, 4)),
            raw_features=np.zeros((3, 3)), original_point_index=np.arange(3),
            labels={"seg_target": truth, "coatjava_seg_pred": truth})
        self.cohort = SimpleNamespace(sha256="fixture-cohort", roles={"test": [0]}, event=lambda role, i: event)
        save_predictions(self.root / "predictions/event_000000.npz", event, truth, np.arange(3),
                         method="fixture", cohort_sha256=self.cohort.sha256)
        metric = self.root / "track_finding_metrics.py"
        metric.write_text('''def event_track_metrics(truth, prediction):
    return {"n_points": len(truth), "agreement": float((truth == prediction).mean())}

def summarize_event_metrics(rows):
    return {"n_points": sum(r["n_points"] for r in rows),
            "agreement": sum(r["agreement"] for r in rows) / len(rows)}
''')
        self.write("provenance.json", {"cohort_sha256": self.cohort.sha256,
                   "metric_sha256": file_sha256(metric), "method": "fixture"})
        self.write("selected_checkpoints.json", [])
        self.write("summary.json", {key: {"n_points": 3, "agreement": 1.} for key in ("metrics", "coatjava")})
        self.row = {"identity": identity, "n_points": 3, "agreement": 1.}
        self.write("per_event_metrics.jsonl", self.row)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value) + "\n")

    def run_audit(self):
        argv = ["audit", "--run", str(self.root), "--cohort", "fixture", "--output", str(self.root / "audit.json")]
        with patch.object(self.audit, "Cohort", return_value=self.cohort), patch("sys.argv", argv), redirect_stdout(StringIO()):
            self.audit.main()

    def test_verified_predictions_and_per_event_cache_pass(self):
        self.run_audit()
        report = json.loads((self.root / "audit.json").read_text())
        self.assertTrue(report["passed"])
        self.assertTrue(report["per_event_metrics_recomputed"])

    def test_modified_bootstrap_input_or_identity_fails(self):
        for row in (dict(self.row, agreement=0.), dict(self.row, identity={"source_file": "other.hipo", "event": 7})):
            with self.subTest(row=row):
                self.write("per_event_metrics.jsonl", row)
                with self.assertRaisesRegex(ValueError, "Per-event metrics or identity"):
                    self.run_audit()
        self.assertFalse((self.root / "audit.json").exists())

    def test_wrong_provenance_fails_even_with_optimized_python(self):
        self.write("provenance.json", {"cohort_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "Cohort fingerprint"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
