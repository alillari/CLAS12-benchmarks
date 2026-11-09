import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import shutil

import h5py
import numpy as np
from mmap_ninja import RaggedMmap

from hdf5_fixture import make_publication
from clas12_benchmarks.data import Cohort, event_key, file_sha256, eggnet_features
from clas12_benchmarks.prepare_cohort import prepare
from clas12_benchmarks.prepare_label_sweep import prepare as sweep
from clas12_benchmarks.predictions import save_predictions
from clas12_benchmarks.sequences import event_sequence, CachedSequence
from clas12_benchmarks.hdf5 import ClusterSource


class CohortTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.base=Path(tmp.name);self.root=make_publication(self.base/"raw")

    def prepare(self,name="cohort",**kwargs):
        return prepare(self.root,self.base/name,reader="hdf5",train=2,val=1,test=1,seed=42,**kwargs)

    def test_manual_inputs_masks_identity_and_repeatability(self):
        with self.prepare() as cohort, self.prepare("again") as again:
            self.assertEqual(cohort.sha256,again.sha256)
            e=cohort.event("train",0)
            expected=np.array([[1,1,0,.2],[1,0,1,.4],[1,-1,0,.6],[1,0,-1,.8],[1,1,0,1]],np.float32)
            np.testing.assert_allclose(e.positions,expected,rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(e.original_point_index,np.arange(5))
            self.assertEqual(e.identity["native_event"],e.identity["original_ordinal"]+1)
            self.assertEqual(e.identity["source_split"],"train")
            for role,split in [("val","validation"),("test","test")]:
                self.assertEqual(cohort.event(role,0).identity["source_split"],split)
            lazy=event_sequence(cohort,"train",cache_size=1)
            self.assertIsInstance(lazy,CachedSequence)
            self.assertEqual(event_key(lazy[0].identity),event_key(e.identity))
            lazy[1];self.assertLessEqual(len(lazy.cache),1)
            save_predictions(self.base/"predictions.npz",e,[3,2,1,0,-1],[4,3,2,1,0],method="test",cohort_sha256=cohort.sha256)
            with np.load(self.base/"predictions.npz") as result:
                self.assertEqual(json.loads(str(result["metadata"]))["schema"],"clas12_baseline_predictions_v2")
                np.testing.assert_array_equal(result["predictions"],[-1,0,1,2,3])

    def test_extension_and_nested_budgets(self):
        with self.prepare() as parent:
            oldtrain=[event_key(parent.event("train",i).identity) for i in range(2)]
            oldtest=event_key(parent.event("test",0).identity)
        with prepare(self.root,self.base/"extended",train=2,val=2,test=2,seed=9,reader="hdf5",extend_from=self.base/"cohort") as extended:
            self.assertEqual(oldtrain,[event_key(extended.event("train",i).identity) for i in range(2)])
            self.assertNotIn(oldtest,[event_key(extended.event("test",i).identity) for i in range(2)])
        report=sweep(self.root,self.base/"cohort",self.base/"sweep",budgets=(1,2,4),seed=7)
        previous=set()
        for budget in [1,2,4]:
            with Cohort(self.base/f"sweep/cohort_{budget}.json") as c:
                current={event_key(c.event("train",i).identity) for i in range(budget)}
                self.assertTrue(previous<=current);previous=current
                self.assertEqual(report["budgets"][str(budget)]["n_training_points"],5*budget)

    def test_relocated_extension_and_sweep_reject_changed_payload(self):
        self.prepare().close()
        copied=self.base/"relocated";shutil.copytree(self.root,copied)
        # Change labels only: geometry normalization and all publication metadata
        # remain identical, so metadata/scale checks alone cannot catch this.
        for path in (copied/"train").glob("*.h5"):
            with h5py.File(path,"r+") as source:
                source["clusters/primary_pid"][:]*=-1
        with self.assertRaisesRegex(ValueError,"Parent selected source content changed"):
            prepare(copied,self.base/"extended",reader="hdf5",train=2,val=1,test=1,
                    extend_from=self.base/"cohort")
        with self.assertRaisesRegex(ValueError,"Parent selected source content changed"):
            sweep(copied,self.base/"cohort",self.base/"sweep",budgets=(1,2,4))
        self.assertFalse((self.base/"extended").exists())
        self.assertFalse((self.base/"sweep").exists())

    def test_identical_relocation_preserves_extension_inputs(self):
        with self.prepare() as parent:
            before=[parent.event("train",i).raw_features for i in range(2)]
        copied=self.base/"relocated";shutil.copytree(self.root,copied)
        with prepare(copied,self.base/"extended",reader="hdf5",train=2,val=1,test=1,
                     extend_from=self.base/"cohort") as extended:
            for i,expected in enumerate(before):
                np.testing.assert_array_equal(extended.event("train",i).raw_features,expected)

    def test_metadata_payload_and_identity_drift_fail(self):
        self.prepare().close()
        specpath=self.base/"cohort/cohort.json";spec=json.loads(specpath.read_text())
        spec["roles"]["train"].append(spec["roles"]["train"][0])
        bad=self.base/"bad.json";bad.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError,"Duplicate"): Cohort(bad)
        with h5py.File(self.root/"train/part-0000.h5","r+") as f:
            f["clusters/xyz"][:,0]+=0.5
        with h5py.File(self.root/"train/part-0002.h5","r+") as f:
            f["clusters/xyz"][:,0]+=0.5
        with self.assertRaisesRegex(ValueError,"content drift"): Cohort(specpath)

    def test_failed_preparation_has_no_published_directory(self):
        with self.assertRaises(ValueError):
            prepare(self.root,self.base/"bad",train=99,val=1,test=1,reader="hdf5")
        self.assertFalse((self.base/"bad").exists())

    def test_validation_coordinates_do_not_fit_training_scales(self):
        with self.prepare() as c: before=c.dataset_spec("train")["radius_scale"]
        for path in (self.root/"validation").glob("*.h5"):
            with h5py.File(path,"r+") as f:f["clusters/xyz"][:]*=1000
        with self.prepare("new") as c:
            self.assertEqual(c.dataset_spec("train")["radius_scale"],before)
            self.assertGreater(c.event("val",0).positions[:,-1].min(),100)

    def test_failure_during_publication_cleans_partial_output(self):
        original=os.link;calls=0
        def fail(source,destination):
            nonlocal calls
            calls+=1
            if calls==2:raise OSError("injected publication failure")
            return original(source,destination)
        with patch("clas12_benchmarks.hdf5_cohort.os.link",side_effect=fail):
            with self.assertRaisesRegex(OSError,"injected"):self.prepare("failed")
        self.assertFalse((self.base/"failed").exists())
        with self.prepare("failed") as cohort:self.assertEqual(len(cohort.roles["train"]),2)

    def test_copied_source_alias_cannot_duplicate_an_event(self):
        with self.prepare() as cohort:spec=json.loads(cohort.path.read_text())
        copied=self.base/"copied";shutil.copytree(self.root,copied)
        spec["datasets"]["alias"]=dict(spec["datasets"]["train"],root=str(copied))
        spec["roles"]["train"].append(dict(spec["roles"]["train"][0],dataset="alias"))
        with ClusterSource(copied,"train") as source:
            spec["datasets"]["alias"]["selected_content_sha256"]=source.content_digest(
                [spec["roles"]["train"][-1]["index"]])
        path=self.base/"duplicate.json";path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError,"Duplicate"):Cohort(path)

    def test_equivalent_logical_event_in_both_backends(self):
        with self.prepare() as cohort:
            event=cohort.event("train",0)
            scales={key:cohort.dataset_spec("train")[key] for key in ("eta_scale","radius_scale")}
        legacy=self.base/"legacy";legacy.mkdir()
        (legacy/"metadata.json").write_text('{}')
        for name,values in {"features":event.raw_features,**event.labels}.items():
            RaggedMmap.from_generator(legacy/f"{name}_pretrain",(values.copy() for _ in range(3)),batch_size=3)
        samples=legacy/"samples_pretrain.jsonl"
        samples.write_text(''.join(json.dumps({"source_file":"logical.hipo","event":i,"sample_mode":"event"})+'\n' for i in range(3)))
        spec={"schema":"clas12_baseline_cohort_v1","datasets":{"logical":{
            "root":str(legacy),"source_split":"pretrain","input_layout":"xyz",**scales,
            "metadata_sha256":file_sha256(legacy/"metadata.json"),"samples_sha256":file_sha256(samples),
            "extra_targets":["coatjava_seg_pred","noise_target","pid_target"]}},
            "roles":{role:[{"dataset":"logical","index":i}] for i,role in enumerate(("train","val","test"))}}
        path=legacy/"cohort.json";path.write_text(json.dumps(spec))
        with Cohort(path) as cohort:
            other=cohort.event("train",0)
            np.testing.assert_array_equal(other.raw_features,event.raw_features)
            np.testing.assert_array_equal(other.positions,event.positions)
            np.testing.assert_array_equal(other.original_point_index,event.original_point_index)
            np.testing.assert_array_equal(eggnet_features(other),eggnet_features(event))
            for field in event.labels:np.testing.assert_array_equal(other.labels[field],event.labels[field])

    def test_truth_and_reconstruction_changes_cannot_change_measured_features(self):
        with self.prepare() as cohort:before=cohort.event("train",0)
        for path in (self.root/"train").glob("*.h5"):
            with h5py.File(path,"r+") as f:
                valid=f["clusters/primary_pid_valid"][:]
                f["clusters/primary_pid"][:]=np.where(valid,2212,0)
                f["clusters/reco_segment"][:]=-1
                f["particles/observed_momentum"][:]*=1000
        with self.prepare("changed") as cohort:after=cohort.event("train",0)
        np.testing.assert_array_equal(before.raw_features,after.raw_features)
        np.testing.assert_array_equal(before.positions,after.positions)
        np.testing.assert_array_equal(eggnet_features(before),eggnet_features(after))
        self.assertFalse(np.array_equal(before.labels["pid_target"],after.labels["pid_target"]))
