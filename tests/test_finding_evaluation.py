import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clas12_benchmarks.data import Event
from clas12_benchmarks.finding_evaluation import (
    FindingEvaluation, MC_PROFILE, PopulationFingerprint, contract, evaluate_events,
    finding_fingerprints, finding_sequence, reference_receipt, require_same_contract,
    require_same_population, verify_evaluation_metadata,
)
from clas12_benchmarks.finding_statistics import mc_intervals
from clas12_benchmarks.fm_reference import track_finding_metrics as fm
from clas12_benchmarks.hdf5_cohort import prepare

from finding_fixture import make_finding_publication


def event(truth, coatjava=None, identity=None):
    truth = np.asarray(truth, dtype=np.int64)
    n = len(truth)
    labels = {"seg_target": truth, "noise_target": (truth < 0).astype(np.int64),
              "pid_target": np.full(n, 11, dtype=np.int64)}
    if coatjava is not None:
        labels["coatjava_seg_pred"] = np.asarray(coatjava, dtype=np.int64)
    return Event(identity or {"source_file": "fixture.hipo", "event": 0},
                 np.ones((n, 3), np.float32), np.ones((n, 4), np.float32), None,
                 labels, np.arange(n, dtype=np.int64))


class FindingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.settings = contract(new_run=True)
        self.evaluation = FindingEvaluation(settings=self.settings)

    def test_scorer_executes_hashed_source_even_when_timestamp_cache_matches(self):
        import os
        import py_compile
        from clas12_benchmarks.finding_evaluation import load_metrics
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"metrics.py"
            path.write_text("VALUE = 1\n")
            stamp=path.stat()
            cache=Path(py_compile.compile(str(path),doraise=True))
            before=cache.read_bytes()
            # Timestamp-based pycs cannot distinguish equal-size edits within
            # one clock tick. Scientific provenance must follow source bytes.
            path.write_text("VALUE = 2\n")
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            module=load_metrics(path)
            self.assertEqual(module.VALUE,2)
            self.assertEqual(cache.read_bytes(),before)

    def test_fixed_full_event_eligibility_and_no_noise_pid_or_prediction_mutation(self):
        truth = [0]+[17]*2+[29]*3+[101]*4+[8]*5+[-1]
        e = event(truth, truth)
        saved = {k: v.copy() for k,v in e.labels.items()}
        projected = self.evaluation.prepare_event(e)
        np.testing.assert_array_equal(projected.labels["seg_target"], [-1]*6+[101]*4+[8]*5+[-1])
        np.testing.assert_array_equal(self.evaluation.prepare_event(projected).labels["seg_target"], projected.labels["seg_target"])
        for key, value in saved.items():
            np.testing.assert_array_equal(e.labels[key], value)
            if key != "seg_target":
                np.testing.assert_array_equal(projected.labels[key], value)
        self.assertIs(FindingEvaluation().prepare_event(e), e)

    def test_legacy_and_new_truth_have_different_denominators(self):
        e = event([3]*4+[9]*3+[-1]*2)
        pred = np.array([7]*4+[-1]*5)
        self.assertEqual(FindingEvaluation().event_metrics(e, pred)["track_efficiency"], .5)
        r = self.evaluation.event_metrics(e, pred)
        self.assertEqual(r["track_efficiency"], 1)
        self.assertEqual(r["n_truth_tracks_below_policy_min_clusters"], 0)

    def test_non_target_predictions_contamination_splits_merges_and_misses(self):
        cases = (([0]*4+[-1]*4, [7]*4+[8]*4, 1., 1., 1.),
                 ([0]*4+[-1]*4, [7]*8, 2/3, 1., .5),
                 ([0]*4, [7,7,8,8], 2/3, .5, 1.),
                 ([0]*4+[1]*4, [7]*8, 1/3, .5, .5),
                 ([0]*4, [-1]*4, 0., 0., None))
        for truth,pred,dice,completeness,purity in cases:
            with self.subTest(truth=truth, pred=pred):
                s = self.evaluation.summarize([self.evaluation.event_metrics(event(truth), np.asarray(pred))])
                self.assertAlmostEqual(s["mc_track_dice_mean"], dice)
                self.assertEqual(s["mc_hit_completeness_global"], completeness)
                self.assertEqual(s["mc_hit_purity_global"], purity)
                self.assertIsNone(s["mc_bg_contamination_global"])
        missed = self.evaluation.summarize([self.evaluation.event_metrics(event([0]*4), np.full(4,-1))])
        self.assertEqual(missed["mc_hit_f1_global"], 0)
        empty = self.evaluation.summarize([])
        self.assertIsNone(empty["mc_track_dice_mean"])
        self.assertIsNone(empty["mc_hit_f1_global"])

    def test_quality_cuts_use_independent_accepted_matching(self):
        e = event([0]*4+[-1]*2)
        pred = np.array([7]*3+[-1]+[7]*2)
        strict = FindingEvaluation(settings={**self.settings, "matching": {"min_purity": .9}})
        ordinary = self.evaluation.event_metrics(e, pred)
        clean = strict.event_metrics(e, pred)
        self.assertEqual(ordinary["n_matched_tracks"], 1)
        self.assertEqual(clean["n_matched_tracks"], 0)
        self.assertEqual(clean["dice_matches"], ordinary["dice_matches"])
        legacy=FindingEvaluation(settings={**strict.contract,"profile":"legacy"})
        self.assertEqual(legacy.event_metrics(e,pred)["n_matched_tracks"],0)

    def test_provenance_join_and_fingerprints_do_not_change_archived_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = make_finding_publication(root / "raw")
            with prepare(raw, root/"cohort", 1,1,1,42) as cohort:
                original = cohort.data_fingerprints()
                e = finding_sequence(cohort, "test", self.evaluation)[0]
                self.assertEqual(len(e.raw_features), 9)
                self.assertEqual((e.labels["seg_target"] >= 0).sum(), 4)
                row = self.evaluation.event_metrics(e, np.full(9,7))
                summary = self.evaluation.summarize([row])
                self.assertEqual(summary["mc_bg_contamination_global"], 1/9)
                self.assertEqual(summary["mc_nontarget_mc_only_fraction_global"], 3/9)
                before = finding_fingerprints(cohort, self.evaluation)
                np.testing.assert_array_equal(cohort.event("test",0).labels["seg_target"], [3]*4+[9]*3+[-1]*2)
                import h5py
                # Change one valid scalar provenance value, preserving raw labels
                # and all bytes covered by the historical selected-content hash.
                member=cohort.roles["test"][0]
                source=cohort.sources[member["dataset"]]
                part,local=source.locate(member["index"])
                path=source.root/source.parts[part][0]
                cohort.close()
                with h5py.File(path, "r+") as file:
                    file["clusters/purity"][local*9] = .75
                self.assertEqual(cohort.data_fingerprints(), original)
                self.assertNotEqual(finding_fingerprints(cohort, self.evaluation), before)

    def test_missing_provenance_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = make_finding_publication(root/"raw", provenance=False)
            with prepare(raw, root/"cohort", 1,1,1,42) as cohort:
                e = finding_sequence(cohort, "test", self.evaluation)[0]
                s = self.evaluation.summarize([self.evaluation.event_metrics(e,np.full(9,2))])
                self.assertIsNone(s["mc_bg_contamination_global"])
                self.assertEqual(s["provenance_coverage_global"], 0)

    def test_legacy_ragged_cohorts_receive_the_same_finding_only_projection(self):
        from mmap_ninja import RaggedMmap
        from clas12_benchmarks.prepare_cohort import prepare as prepare_ragged
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/"raw";raw.mkdir()
            (raw/"metadata.json").write_text("{}")
            xyz=np.column_stack((np.arange(9)+1,np.ones(9),np.arange(9))).astype("f4")
            truth=np.array([3]*4+[9]*3+[-1]*2)
            for split in ("pretrain","test"):
                for field,value in (("features",xyz),("seg_target",truth),("coatjava_seg_pred",truth)):
                    RaggedMmap.from_generator(raw/f"{field}_{split}",(value.copy() for _ in range(4)),batch_size=4)
                (raw/f"samples_{split}.jsonl").write_text("".join(json.dumps({"source_file":split+".hipo", "event":i,
                                                                          "sample_mode":"event"})+"\n" for i in range(4)))
            with prepare_ragged(raw,root/"cohort",train=1,val=1,test=1) as cohort:
                before=cohort.data_fingerprints()
                for role in ("train","val","test"):
                    projected=finding_sequence(cohort,role,self.evaluation)[0]
                    np.testing.assert_array_equal(projected.labels["seg_target"],[3]*4+[-1]*5)
                    np.testing.assert_array_equal(cohort.event(role,0).labels["seg_target"],truth)
                self.assertEqual(cohort.data_fingerprints(),before)

    def test_contract_guards_and_legacy_defaults(self):
        self.assertEqual(contract()["profile"], "legacy")
        for value in ({"truth_policy":"min3"}, {"profile":"dice"}, {"surprise":1},
                      {"matching":{"min_purity":2}}, {"matching":{"background_label":0}}):
            with self.assertRaises(ValueError):
                contract(value)
        metadata = self.evaluation.metadata()
        verify_evaluation_metadata(metadata, self.evaluation)
        for key in metadata:
            bad = dict(metadata); bad.pop(key)
            with self.assertRaises(ValueError):
                verify_evaluation_metadata(bad, self.evaluation)
        with self.assertRaisesRegex(ValueError,"contracts differ"):
            require_same_contract(metadata, {})
        with self.assertRaisesRegex(ValueError,"populations differ"):
            require_same_population({**metadata,"evaluation_population_sha256":"a"}, metadata)
        complete={**metadata,"evaluation_population_sha256":"a"}
        with self.assertRaisesRegex(ValueError,"source fingerprints differ"):
            require_same_population(complete,{**complete,"metrics_source_sha256":"changed"})
        reference_receipt()

    def test_population_uses_physical_identity_truth_and_cluster_membership(self):
        e = event([0]*4)
        def fingerprint(e):
            value=PopulationFingerprint(); value.add(e,self.evaluation); return value.hexdigest()
        before=fingerprint(e)
        e.positions *= 2
        self.assertEqual(fingerprint(e), before)  # per-budget normalization is allowed
        e.raw_features[0,0] *= 2
        self.assertNotEqual(fingerprint(e), before)

    def test_global_bootstrap_uses_counts_and_keeps_undefined_values(self):
        a=self.evaluation.event_metrics(event([0]*4), np.full(4,7))
        b=self.evaluation.event_metrics(event([0]*8+[1]*8+[2]*8,identity={"source_file":"fixture.hipo","event":1}), np.full(24,-1))
        rows=[dict(a, finding_evaluation=self.settings,identity={"event":0}),
              dict(b,finding_evaluation=self.settings,identity={"event":1})]
        s=self.evaluation.summarize(rows)
        self.assertEqual(s["mc_track_dice_mean"], .25)
        self.assertEqual(s["mc_track_dice_event_macro"], .5)
        self.assertEqual(s["mc_hit_completeness_global"], 4/28)
        result=mc_intervals(rows, replicates=128)
        self.assertIsNone(result["intervals_95"]["mc_bg_contamination_global"])
        self.assertEqual(result["defined_resamples"]["mc_track_dice_mean"],128)
        delta=mc_intervals(rows,rows,replicates=32)
        self.assertEqual(delta["intervals_95"]["mc_track_dice_mean"],[0.,0.])
        with self.assertRaises(ValueError):
            mc_intervals([rows[0],dict(rows[1],finding_evaluation=contract())])

    def test_model_and_coatjava_use_same_protocol_and_export_final_labels(self):
        e=event([0]*4+[9]*3+[-1]*2, [1]*4+[2]*3+[-1]*2)
        prediction=np.array([7]*4+[-1]*5)
        with tempfile.TemporaryDirectory() as directory:
            result=evaluate_events([e],[prediction],settings=self.settings,output=directory,method="fixture",cohort_sha="sha")
            self.assertEqual(result["metrics"]["mc_track_dice_mean"],1)
            self.assertEqual(result["coatjava"]["mc_track_dice_mean"],1)
            self.assertEqual(result["coatjava"]["track_purity_global"],.5)
            with np.load(Path(directory)/"predictions/event_000000.npz") as saved:
                np.testing.assert_array_equal(saved["predictions"], prediction)
                self.assertEqual(json.loads(str(saved["metadata"]))["finding_evaluation"],self.settings)


class CommittedFMParityTests(unittest.TestCase):
    """Differential checks against Git objects, ignoring the FM working tree."""
    @classmethod
    def setUpClass(cls):
        import os
        cls.root=Path(os.environ.get("CLAS12_FM_ROOT",Path(__file__).resolve().parents[2]/"CLAS12"))
        if not (cls.root/".git").exists():
            raise unittest.SkipTest("Committed FM source checkout unavailable; pinned/golden tests still run")
        cls.temp=tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        receipt=reference_receipt()
        cls.reference=[]
        for name,item in receipt["files"].items():
            code=subprocess.check_output(["git","-C",str(cls.root),"show",f"{receipt['commit']}:{item['source_path']}"])
            path=Path(cls.temp.name)/name; path.write_bytes(code)
            from clas12_benchmarks.data import file_sha256
            if file_sha256(path) != item["sha256"]:
                raise AssertionError("Packaged helper differs from committed FM source")
            spec=importlib.util.spec_from_file_location("committed_fm_"+path.stem,path)
            module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
            cls.reference.append(module)
        cls.metrics,cls.eligibility=cls.reference

    def test_random_event_and_global_outputs_match_main_fm_exactly(self):
        rng=np.random.default_rng(421)
        evaluation=FindingEvaluation(settings=contract(new_run=True))
        ours,theirs=[],[]
        for _ in range(80):
            n=int(rng.integers(0,85))
            truth,pred=rng.integers(-1,6,(2,n))
            e=event(truth)
            np.testing.assert_array_equal(evaluation.prepare_event(e).labels["seg_target"],
                                          self.eligibility.eligible_primary_labels(truth))
            expected=self.metrics.event_mc_reconstruction_metrics(self.eligibility.eligible_primary_labels(truth),pred,point_ids=e.original_point_index)
            actual=evaluation.event_metrics(e,pred)
            self.assertEqual(actual,expected)
            ours.append(actual); theirs.append(expected)
        actual=evaluation.summarize(ours)
        for key,value in self.metrics.summarize_mc_reconstruction_metrics(theirs).items():
            self.assertEqual(actual[key],value,key)

    def test_saved_assignments_match_the_committed_fm_offline_rescorer(self):
        from types import ModuleType
        from unittest.mock import patch
        receipt=reference_receipt()
        path=Path(self.temp.name)/"rescore_track_finding.py"
        path.write_bytes(subprocess.check_output(["git","-C",str(self.root),"show",
            f"{receipt['commit']}:train/downstream/eval/rescore_track_finding.py"]))
        downstream=ModuleType("train.downstream");downstream.track_finding_metrics=self.metrics
        datasets=ModuleType("fm4npp.datasets");datasets.track_eligibility=self.eligibility
        train=ModuleType("train");train.downstream=downstream
        fm4npp=ModuleType("fm4npp");fm4npp.datasets=datasets
        with patch.dict(sys.modules,{"train":train,"train.downstream":downstream,
                                    "fm4npp":fm4npp,"fm4npp.datasets":datasets}):
            spec=importlib.util.spec_from_file_location("committed_fm_offline",path)
            scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
        truth=np.array([0]*4+[9]*3+[-1]*2)
        prediction=np.array([7]*4+[-1]*4+[7])
        coat=np.array([1]*4+[2]*3+[-1]*2)
        sidecars={"background_only":np.array([False]*8+[True]),
                  "ownership":np.array([1]*7+[3,0],dtype="i1"),
                  "origin_noise":np.array([False]*8+[True]),
                  "purity":np.array([1.]*7+[0.,0.])}
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/"labels.npz"
            np.savez_compressed(archive,truth=truth,pred=prediction,coatjava=coat,
                                offsets=np.array([0,9]),point_ids=np.arange(9),**sidecars)
            expected=scorer.rescore_archive(archive,("pred","coatjava"),truth_policy="primary_mc_min4_v1")
            e=event(truth,coat)
            e.context={"clusters."+key:value for key,value in sidecars.items()}
            actual=evaluate_events([e],[prediction],settings=contract(new_run=True))
            for ours,theirs in (("metrics","pred"),("coatjava","coatjava")):
                for key,value in expected["metrics"][theirs].items():
                    self.assertEqual(actual[ours][key],value,key)


if __name__ == "__main__":
    unittest.main()
