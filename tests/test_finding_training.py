from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from clas12_benchmarks.data import Cohort, file_sha256
from clas12_benchmarks.evaluate_frozen import evaluate_frozen
from clas12_benchmarks.finding_evaluation import DEFAULT_METRICS, contract
from clas12_benchmarks.hdf5_cohort import prepare
from clas12_benchmarks.tracking import run
from clas12_benchmarks.upstream import ROOT

from finding_fixture import make_finding_publication


class FindingTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=make_finding_publication(self.root/"data")
        with prepare(self.data,self.root/"cohort",2,1,1,42):
            pass
        self.cohort=self.root/"cohort/cohort.json"
        config=json.loads((ROOT/"configs/tracking.json").read_text())
        config.update(steps_per_stage=4,validation_interval=2,checkpoint_validation_events=1,warmup_steps=1)
        for key in ("encoder_hidden","node_0_hidden","node_hidden","edge_hidden","decoder_hidden",
                    "node_rep_dim","edge_rep_dim","node_pspace_dim"):
            config["eggnet"][key]=8
        config["eggnet"].update(n_iters=1,n_gnns_per_iter=1,dbscan_eps=[.1,.5],dbscan_min_samples=[2],
                                 checkpoint_dbscan_eps=[.1],checkpoint_dbscan_min_samples=[2])
        config["exatrkx"]["embedding"].update(emb_hidden=8,nb_layer=2,emb_dim=2)
        config["exatrkx"]["filter"].update(hidden=8,nb_layer=2)
        config["exatrkx"].update(gnn_processing_steps=1,gnn_cuts=[.3,.7],checkpoint_gnn_cuts=[.3])
        self.config=self.root/"recipe.json";self.config.write_text(json.dumps(config))

    def train(self,method):
        args=SimpleNamespace(method=method,cohort=str(self.cohort),config=str(self.config),
            metrics=str(DEFAULT_METRICS),output=str(self.root/method),device="cpu",threads=2,smoke=False,resume=False)
        captured=[]
        import clas12_benchmarks.tracking as tracking
        function_name="eggnet_contrastive_loss" if method=="eggnet" else "embedding_loss"
        original=getattr(tracking,function_name)
        def objective(z,labels,*a,**kw):
            captured.append(labels.detach().cpu().numpy().copy())
            return original(z,labels,*a,**kw)
        with patch.object(tracking,function_name,objective),redirect_stdout(StringIO()):
            run(args)
        self.assertTrue(captured)
        self.assertTrue(all(np.array_equal(y,np.array([3]*4+[-1]*5)) for y in captured))
        return Path(args.output),args

    def test_both_training_pipelines_frozen_reload_audit_and_diagnostics(self):
        spec=importlib.util.spec_from_file_location("finding_training_audit",ROOT/"scripts/audit_tracking_predictions.py")
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        for method in ("eggnet","exatrkx"):
            with self.subTest(method=method):
                parent,_=self.train(method)
                summary=json.loads((parent/"summary.json").read_text())
                self.assertEqual(summary["finding_evaluation"],contract(new_run=True))
                self.assertEqual(summary["metrics"]["n_true_tracks"],1)
                self.assertEqual(summary["coatjava"]["n_true_tracks"],1)
                self.assertEqual(summary["metrics"]["n_truth_tracks_below_policy_min_clusters"],0)
                before={str(p.relative_to(parent)):file_sha256(p) for p in parent.rglob("*") if p.is_file()}
                frozen=self.root/(method+"-frozen")
                with redirect_stdout(StringIO()):
                    evaluate_frozen(parent,self.cohort,self.cohort,frozen,device="cpu",threads=2)
                self.assertEqual(json.loads((frozen/"summary.json").read_text()),summary)
                for path in (parent,frozen):
                    with patch("sys.argv",["audit","--run",str(path),"--cohort",str(self.cohort),
                                          "--output",str(self.root/(path.name+"-audit.json"))]),redirect_stdout(StringIO()):
                        audit.main()
                self.assertEqual(before,{str(p.relative_to(parent)):file_sha256(p) for p in parent.rglob("*") if p.is_file()})
                if method=="exatrkx":
                    from clas12_benchmarks.exatrkx_diagnostics import analyze
                    from clas12_benchmarks.exatrkx_threshold_sweep import sweep
                    args=SimpleNamespace(run=str(parent),cohort=str(self.cohort),output=str(self.root/"diag"),
                        split="val",device="cpu",threads=2,max_events=None,radii=[2.],ks=[32])
                    with redirect_stdout(StringIO()):
                        analyze(args)
                    self.assertEqual(json.loads((Path(args.output)/"summary.json").read_text())["actual_gnn"]["n_true_tracks"],1)
                    args.output=str(self.root/"sweep");args.cuts=[.3,.7];args.bootstrap=16
                    with redirect_stdout(StringIO()):
                        sweep(args)
                    report=json.loads((Path(args.output)/"summary.json").read_text())
                    self.assertIn("best_by_mc_dice",report)
                    self.assertEqual(report["finding_evaluation"],summary["finding_evaluation"])

    def test_refused_resume_does_not_replace_the_frozen_evaluator(self):
        parent,args=self.train("eggnet")
        before={str(p.relative_to(parent)):file_sha256(p) for p in parent.rglob("*") if p.is_file()}
        wrong=self.root/"wrong-metrics.py";wrong.write_text("raise AssertionError('must never import')\n")
        args.metrics=str(wrong);args.resume=True
        with self.assertRaisesRegex(ValueError,"snapshot differs"):
            run(args)
        self.assertEqual(before,{str(p.relative_to(parent)):file_sha256(p) for p in parent.rglob("*") if p.is_file()})

    def test_comparison_cli_rescores_historical_supervision_separately(self):
        import subprocess
        import sys
        legacy=json.loads(self.config.read_text());legacy["finding_evaluation"]=contract()
        legacy_path=self.root/"legacy.json";legacy_path.write_text(json.dumps(legacy))
        parent=self.root/"historical"
        with redirect_stdout(StringIO()):
            run(SimpleNamespace(method="eggnet",cohort=str(self.cohort),config=str(legacy_path),
                metrics=str(DEFAULT_METRICS),output=str(parent),device="cpu",threads=2,smoke=False,resume=False))
        before={str(p.relative_to(parent)):file_sha256(p) for p in parent.rglob("*") if p.is_file()}
        output=self.root/"comparison"
        command=[sys.executable,str(ROOT/"scripts/run_tuned_comparison.py"),"--method","eggnet",
                 "--cohort",str(self.cohort),"--original-cohort",str(self.cohort),"--parent",str(parent),
                 "--config",str(self.config),"--metrics",str(DEFAULT_METRICS),"--output",str(output),
                 "--device","cpu","--threads","2"]
        subprocess.run(command,check=True,capture_output=True,text=True)
        report=json.loads((output/"comparison.json").read_text())
        self.assertEqual(report["historical_original_frozen"]["metrics"]["n_true_tracks"],2)
        self.assertEqual(report["original_frozen"]["metrics"]["n_true_tracks"],1)
        self.assertEqual(report["tuned"]["evaluation_population_sha256"],report["original_frozen"]["evaluation_population_sha256"])
        self.assertIn("mc_track_dice_mean",report["paired_bootstrap"]["delta_95_percent_intervals"])
        self.assertTrue((output/"original_rescored/receipt.json").exists())
        self.assertEqual(before,{str(p.relative_to(parent)):file_sha256(p) for p in parent.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
