import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from clas12_benchmarks.finding_evaluation import FindingEvaluation, finding_sequence, contract
from clas12_benchmarks.hdf5 import ClusterSource
from clas12_benchmarks.prepare_finding_cohort import prepare

from finding_fixture import make_finding_publication


class FMFindingCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=make_finding_publication(self.root/"data")
        self.rows={}
        for split in ("train","validation","test"):
            with ClusterSource(self.data,split) as source:
                identities=source.identities([3,1])
                self.rows[split]=[{"source_event_index":i,"source_id":identities[i]["source_id"],
                                   "original_ordinal":identities[i]["original_ordinal"],"segment_label":None} for i in (3,1)]
        self.path=self.root/"cohorts.json"
        self.path.write_text(json.dumps(self.rows))

    def test_exact_membership_order_and_effective_counts(self):
        with prepare(self.data,self.path,self.root/"cohort","fixture-plan") as cohort:
            evaluation=FindingEvaluation(settings=contract(new_run=True))
            for role in ("train","val","test"):
                events=finding_sequence(cohort,role,evaluation)
                self.assertEqual([e.identity["original_ordinal"] for e in events],[3,1])
                for e in events:
                    self.assertEqual(np.unique(e.labels["seg_target"]).tolist(),[-1,3])
                    self.assertEqual(len(e.raw_features),9)
            summary=json.loads((self.root/"cohort/summary.json").read_text())
            self.assertEqual(summary["roles"]["train"]["n_eligible_tracks"],2)
            self.assertEqual(summary["roles"]["train"]["n_clusters"],18)
            with self.assertRaises(FileExistsError):
                prepare(self.data,self.path,self.root/"cohort","fixture-plan")

    def test_plan_identity_duplicate_and_segment_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"plan_id"):
            prepare(self.data,self.path,self.root/"bad-plan","wrong-plan")
        for name,mutation in (("identity",lambda r:r["test"][0].update(original_ordinal=99)),
                              ("duplicate",lambda r:r["test"].append(r["test"][0].copy())),
                              ("segment",lambda r:r["train"][0].update(segment_label=3))):
            with self.subTest(name=name):
                rows=json.loads(json.dumps(self.rows));mutation(rows);self.path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    prepare(self.data,self.path,self.root/name,"fixture-plan")
                self.assertFalse((self.root/name).exists())


if __name__ == "__main__":
    unittest.main()
