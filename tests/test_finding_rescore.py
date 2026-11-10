from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clas12_benchmarks.data import file_sha256
from clas12_benchmarks.finding_evaluation import DEFAULT_METRICS, FindingEvaluation, contract, evaluate_events, finding_sequence, reference_receipt
from clas12_benchmarks.fm_reference import track_finding_metrics as fm
from clas12_benchmarks.hdf5_cohort import prepare
from clas12_benchmarks.prepare_cohort import write_json
from clas12_benchmarks.rescore_finding import rescore

from finding_fixture import make_finding_publication, make_legacy_finding_cohort


def archived_run(root, settings=None, *, backend="hdf5", coatjava=True):
    if backend=="hdf5":
        raw=make_finding_publication(root/"raw")
        cohort=prepare(raw,root/"cohort",1,1,1,42)
    else:
        cohort=make_legacy_finding_cohort(root,coatjava=coatjava)
    parent=root/"parent"; parent.mkdir()
    shutil.copyfile(DEFAULT_METRICS,parent/"track_finding_metrics.py")
    (parent/"embedding_best.pt").write_bytes(b"checkpoint identity; never deserialized by the CPU rescorer")
    write_json(parent/"selected_checkpoints.json",[{"stage":"embedding","selected_step":2,
               "checkpoint_sha256":file_sha256(parent/"embedding_best.pt")}])
    write_json(parent/"frozen_selection.json",{"eps":.2,"min_samples":4})
    evaluation=FindingEvaluation(settings=settings)
    events=finding_sequence(cohort,"test",evaluation)
    evaluate_events(events,[np.array([7]*4+[-1]*5)],settings=settings,output=parent,method="eggnet",cohort_sha=cohort.sha256)
    provenance={"method":"eggnet","config":{},"cohort_sha256":cohort.sha256,
                "metric_sha256":file_sha256(DEFAULT_METRICS),"data_array_sha256":cohort.data_fingerprints()}
    if settings is not None:
        from clas12_benchmarks.finding_evaluation import finding_fingerprints
        provenance.update(evaluation.metadata(),finding_data_sha256=finding_fingerprints(cohort,evaluation),fm_reference=reference_receipt())
        provenance["config"]["finding_evaluation"]=settings
    write_json(parent/"provenance.json",provenance)
    cohort.close()
    return parent,root/"cohort/cohort.json"


class FindingRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.parent,self.cohort=archived_run(self.root)

    def test_rescore_preserves_every_parent_byte_and_records_new_truth(self):
        before={str(p.relative_to(self.parent)):file_sha256(p) for p in self.parent.rglob("*") if p.is_file()}
        result=rescore(self.parent,self.cohort,self.root/"rescored")
        self.assertEqual(result["metrics"]["mc_track_dice_mean"],1)
        self.assertEqual(result["metrics"]["track_efficiency_global"],1)
        self.assertEqual(json.loads((self.parent/"summary.json").read_text())["metrics"]["track_efficiency_global"],.5)
        receipt=json.loads((self.root/"rescored/receipt.json").read_text())
        self.assertEqual(receipt["n_relabelled_truth_points"],3)
        self.assertFalse(receipt["checkpoint_selection_modified"])
        self.assertFalse(receipt["thresholds_recalibrated"])
        self.assertEqual(file_sha256(self.root/"rescored/track_finding_metrics.py"),receipt["metrics_source_sha256"])
        self.assertEqual(file_sha256(self.root/"rescored/track_eligibility.py"),
                         receipt["fm_reference"]["files"]["track_eligibility.py"]["sha256"])
        self.assertIn("finding_evaluation.py",receipt["evaluation_source_sha256"])
        after={str(p.relative_to(self.parent)):file_sha256(p) for p in self.parent.rglob("*") if p.is_file()}
        self.assertEqual(before,after)
        self.assertFalse((self.root/"rescored/predictions").exists())

    def test_as_saved_is_explicit_and_preserves_historical_denominator(self):
        result=rescore(self.parent,self.cohort,self.root/"historical",settings={"profile":"mc_reconstruction_v1","truth_policy":"as_saved"})
        self.assertEqual(result["metrics"]["n_true_tracks"],2)
        self.assertEqual(result["metrics"]["mc_track_dice_mean"],.5)
        self.assertEqual(result["metrics"]["n_truth_tracks_below_policy_min_clusters"],1)
        self.assertIsNone(result["track_eligibility_contract"])

    def test_outputs_cannot_replace_or_nest_under_an_archive(self):
        for output in (self.parent,self.parent/"rescored"):
            with self.assertRaisesRegex(ValueError,"outside"):
                rescore(self.parent,self.cohort,output)
        output=self.root/"existing";output.mkdir();(output/"keep").write_text("keep")
        with self.assertRaises(FileExistsError):
            rescore(self.parent,self.cohort,output)
        self.assertEqual((output/"keep").read_text(),"keep")

    def test_identity_alignment_checkpoint_and_raw_source_corruption_fail(self):
        path=self.parent/"predictions/event_000000.npz"
        original=path.read_bytes()
        with np.load(path,allow_pickle=False) as file:
            saved={k:file[k] for k in file.files}
        for name,value in (("predictions",np.zeros(8,dtype=int)),("original_point_index",np.zeros(9,dtype=int))):
            with self.subTest(name=name):
                np.savez_compressed(path,**{**saved,name:value})
                with self.assertRaises(ValueError):
                    rescore(self.parent,self.cohort,self.root/name)
                path.write_bytes(original)
        metadata=json.loads(str(saved["metadata"])); metadata["identity"]["original_ordinal"]+=1
        np.savez_compressed(path,**{**saved,"metadata":json.dumps(metadata)})
        with self.assertRaisesRegex(ValueError,"identity"):
            rescore(self.parent,self.cohort,self.root/"identity")
        path.write_bytes(original)
        checkpoint=(self.parent/"embedding_best.pt").read_bytes()
        (self.parent/"embedding_best.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            rescore(self.parent,self.cohort,self.root/"checkpoint")
        (self.parent/"embedding_best.pt").write_bytes(checkpoint)
        from clas12_benchmarks.data import Cohort
        import h5py
        with Cohort(self.cohort) as cohort:
            member=cohort.roles["test"][0];source=cohort.sources[member["dataset"]]
            part,local=source.locate(member["index"]);raw=source.root/source.parts[part][0]
        with h5py.File(raw,"r+") as file:
            file["clusters/xyz"][local*9,0] += 1
        with self.assertRaisesRegex(ValueError,"content drift"):
            rescore(self.parent,self.cohort,self.root/"raw-source")

    def test_changed_provenance_is_rejected_for_a_versioned_parent(self):
        import h5py
        from clas12_benchmarks.data import Cohort
        root=self.root/"versioned";root.mkdir()
        parent,path=archived_run(root,contract(new_run=True))
        with Cohort(path) as cohort:
            member=cohort.roles["test"][0];source=cohort.sources[member["dataset"]]
            part,local=source.locate(member["index"]);raw=source.root/source.parts[part][0]
        with h5py.File(raw,"r+") as file:
            file["clusters/purity"][local*9] = .75
        with self.assertRaisesRegex(ValueError,"sidecars changed"):
            rescore(parent,path,root/"rejected")
        self.assertFalse((root/"rejected").exists())

    def test_current_profile_results_can_be_rescored_and_audited(self):
        import importlib.util
        from unittest.mock import patch
        root=self.root/"current";root.mkdir()
        parent,cohort=archived_run(root,contract(new_run=True))
        spec=importlib.util.spec_from_file_location("new_finding_audit",Path(__file__).resolve().parents[1]/"scripts/audit_tracking_predictions.py")
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        def run_audit():
            with patch("sys.argv",["audit","--run",str(parent),"--cohort",str(cohort),"--output",str(root/"audit.json")]),redirect_stdout(StringIO()):
                audit.main()
        run_audit()
        rescored=rescore(parent,cohort,root/"rescored")
        self.assertEqual(rescored,json.loads((parent/"summary.json").read_text()))
        summary=json.loads((parent/"summary.json").read_text());summary.pop("track_eligibility_contract")
        write_json(parent/"summary.json",summary)
        with self.assertRaisesRegex(ValueError,"metadata mismatch"):
            run_audit()

    def test_cli_and_import_do_not_require_torch(self):
        env=__import__("os").environ.copy()
        command=[sys.executable,"-m","clas12_benchmarks.rescore_finding","--run",str(self.parent),
                 "--cohort",str(self.cohort),"--output",str(self.root/"cli")]
        subprocess.run(command,check=True,capture_output=True,text=True,env=env)
        result=subprocess.run([sys.executable,"-c","import sys; import clas12_benchmarks.rescore_finding; assert 'torch' not in sys.modules"],
                              check=True,capture_output=True,text=True,env=env)
        self.assertEqual(result.returncode,0)

    def test_reading_archived_scorer_does_not_create_bytecode_in_parent(self):
        import os
        env=os.environ.copy()
        env.pop("PYTHONDONTWRITEBYTECODE",None)
        env.pop("PYTHONPYCACHEPREFIX",None)
        before={str(p.relative_to(self.parent)):file_sha256(p) for p in self.parent.rglob("*") if p.is_file()}
        subprocess.run([sys.executable,"-m","clas12_benchmarks.rescore_finding",
            "--run",str(self.parent),"--cohort",str(self.cohort),"--output",str(self.root/"cache-enabled"),
            "--metrics",str(self.parent/"track_finding_metrics.py")],
            env=env,check=True,capture_output=True,text=True)
        after={str(p.relative_to(self.parent)):file_sha256(p) for p in self.parent.rglob("*") if p.is_file()}
        self.assertEqual(before,after)

    def test_output_cannot_change_a_materialized_legacy_cohort_fingerprint(self):
        from clas12_benchmarks.data import Cohort
        root=self.root/"materialized";root.mkdir()
        parent,path=archived_run(root,backend="ragged_mmap")
        output=path.parent/"rescored"
        with Cohort(path) as cohort:
            before=cohort.data_fingerprints()
        with self.assertRaisesRegex(ValueError,"cohort"):
            rescore(parent,path,output)
        self.assertFalse(output.exists())
        with Cohort(path) as cohort:
            self.assertEqual(cohort.data_fingerprints(),before)

    def test_model_only_legacy_results_rescore_without_a_coatjava_reference(self):
        root=self.root/"model-only";root.mkdir()
        parent,path=archived_run(root,backend="ragged_mmap",coatjava=False)
        result=rescore(parent,path,root/"rescored")
        self.assertEqual(result["metrics"]["mc_track_dice_mean"],1)
        self.assertEqual(result["coatjava"]["n_events"],0)
        self.assertNotIn("model_minus_coatjava",result)
        self.assertEqual((root/"rescored/per_event_coatjava_metrics.jsonl").read_text(),"")
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"scripts/audit_tracking_predictions.py"),
            "--run",str(parent),"--cohort",str(path),"--output",str(root/"audit.json")],
            check=True,capture_output=True,text=True)


if __name__ == "__main__":
    unittest.main()
