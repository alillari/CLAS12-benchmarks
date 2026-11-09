import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from hdf5_fixture import make_publication
from clas12_benchmarks.prepare_cohort import prepare
from clas12_benchmarks.hdf5_momentum import build_manifest
from clas12_benchmarks.momentum_manifest import TrackManifest
from clas12_benchmarks.momentum_evaluation import evaluate, verify_native_pt, verify_score_tables


class MomentumHDF5Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.base=Path(tmp.name);self.root=make_publication(self.base/"raw")
        with prepare(self.root,self.base/"cohort",reader="hdf5",train=2,val=1,test=1,seed=42): pass
        self.pack=self.base/"tracks.npz"

    def build(self,**kwargs):
        build_manifest(self.base/"cohort/cohort.json",self.pack,min_clusters=1,max_clusters=100,
                       train_track_budgets=(1,3),**kwargs)
        return TrackManifest(self.pack)

    def write_predictions(self,m,rows,**changes):
        path=self.base/"pred.csv"
        records=[]
        for i in reversed(rows):
            record=dict(zip(m.identity_fields,m.identity(i)))
            record.update(schema="clas12_momentum_predictions_v2",momentum_unit="GeV",manifest_sha256=m.sha256)
            record.update({f"pred_{axis}_gev":value for axis,value in zip(("px","py","pz"),m.truth[i])})
            record.update(changes);records.append(record)
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        return path

    def test_tid_join_units_point_membership_and_keyed_scoring(self):
        m=self.build();rows=m.rows("evaluation")
        self.assertEqual(len(rows),2)
        np.testing.assert_allclose(m.truth[rows[0]],[.3,-.4,1.2],rtol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(m.truth[rows[0]])),1.3,places=6)
        self.assertEqual(m.identity(rows[0])[-1],17)
        np.testing.assert_array_equal(m.hits(rows[0])[1],[0,1])
        self.assertEqual(m.rows("train_3").tolist(),[0,1,2])
        result=evaluate(m,self.write_predictions(m,rows))
        self.assertEqual(result["all_tracks"]["p_mae_mev"],0)
        self.assertEqual(result["cvt_valid"]["model"]["n"],2)
        for change in [dict(momentum_unit="MeV"),dict(manifest_sha256="wrong")]:
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                evaluate(m,self.write_predictions(m,rows,**change))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            evaluate(m,self.write_predictions(m,[*rows,rows[0]]))
        with self.assertRaisesRegex(ValueError,"Missing"):
            evaluate(m,self.write_predictions(m,rows[:1]))
        with self.assertRaisesRegex(ValueError,"companion"): evaluate(m,self.base/"absent.csv",companion=True)

    def test_invalid_truth_and_missing_reference_keep_denominators(self):
        # These are additional consumed fields, beyond the tracking cohort hash.
        for path in (self.root/"test").glob("*.h5"):
            with h5py.File(path,"r+") as f:
                f["particles/observed_valid"][::2,0]=False
                f["reco/cvt_tracks/unique_owner"][:]=False
        m=self.build();rows=m.rows("evaluation")
        self.assertEqual(len(rows),1)
        self.assertTrue(np.isnan(m.cvt[rows]).all())
        result=evaluate(m,self.write_predictions(m,rows))
        self.assertEqual(result["all_tracks"]["n"],1)
        self.assertEqual(result["cvt_valid"]["model"],{"n":0})
        self.assertEqual(m.meta["selection_counts"]["evaluation"]["invalid_truth"],1)

    def test_changed_particle_membership_invalidates_frozen_cohort(self):
        for path in (self.root/"train").glob("*.h5"):
            with h5py.File(path,"r+") as f: f["particles/tid"][::2]=17
        with self.assertRaisesRegex(ValueError,"content drift"): self.build()
        self.assertFalse(self.pack.exists())

    def test_pack_array_tamper_and_existing_output_rejected(self):
        m=self.build()
        with self.assertRaises(FileExistsError): self.build()
        with np.load(self.pack) as source: arrays={key:source[key] for key in source.files}
        arrays["truth_gev"][0,0]*=1000
        np.savez(self.base/"tampered.npz",**arrays)
        with self.assertRaisesRegex(ValueError,"content mismatch"): TrackManifest(self.base/"tampered.npz")

    def test_reconstruction_pass_changes_reference_not_truth(self):
        first=self.build()
        self.pack=self.base/"cvtrec.npz"
        second=self.build(cvt_pass="cvtrec")
        np.testing.assert_array_equal(first.truth,second.truth)
        np.testing.assert_array_equal(first.point_index,second.point_index)
        self.assertAlmostEqual(float(np.linalg.norm(second.cvt[0])),2.6,places=6)
        self.assertNotEqual(first.meta["consumed_content_sha256"],second.meta["consumed_content_sha256"])

    def test_legacy_score_references_cannot_verify_hdf5_targets(self):
        manifest=self.build()
        result=evaluate(manifest,self.write_predictions(manifest,manifest.rows("evaluation")))
        with self.assertRaisesRegex(ValueError,"Legacy reference tables"):
            verify_score_tables(result,self.base/"absent","model")
        with self.assertRaisesRegex(ValueError,"Legacy reference summaries"):
            verify_native_pt(result,self.base/"absent.json")

    def test_ambiguous_reference_keeps_all_model_tracks(self):
        for path in (self.root/"test").glob("*.h5"):
            with h5py.File(path,"r+") as f:
                f["reco/contributors/tid"][1::4]=17
                f["reco/cvt_tracks/purity_tid"][1::2]=17
        m=self.build();rows=m.rows("evaluation")
        self.assertEqual(len(rows),2)
        self.assertTrue(np.isnan(m.cvt[rows]).all())
        self.assertEqual(m.meta["selection_counts"]["evaluation"]["cvt_ambiguous"],1)

    def test_declared_cluster_cuts_and_empty_populations(self):
        build_manifest(self.base/"cohort/cohort.json",self.pack,min_clusters=2,max_clusters=2)
        m=TrackManifest(self.pack)
        self.assertEqual(len(m.rows("evaluation")),1)
        self.assertEqual(m.meta["selection_counts"]["evaluation"]["length_excluded"],1)
        empty=self.base/"empty.npz"
        build_manifest(self.base/"cohort/cohort.json",empty,min_clusters=12,max_clusters=100)
        e=TrackManifest(empty)
        self.assertEqual(len(e.rows("train")),0)
        self.assertEqual(e.truth.shape,(0,3))
