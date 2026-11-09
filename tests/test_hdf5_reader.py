import json
import multiprocessing
import os
import pickle
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from hdf5_fixture import make_publication
from clas12_benchmarks.hdf5 import ClusterSource, POINT_FIELDS, cluster_projection, NATIVE_NOISE_POLICY


def worker_read(source, queue):
    value = source.read(4, ["events.original_ordinal"])["events.original_ordinal"]
    queue.put((int(value), source._pid == os.getpid(), len(source._files)))
    source.close()


class ReaderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = make_publication(Path(tmp.name)/"raw")

    def test_boundaries_own_offsets_and_handles(self):
        with ClusterSource(self.root,"train",max_open=1) as source:
            self.assertEqual(len(source),5)
            for index in [4,0,2,1,3,-1]:
                rows=source.read(index,["events.original_ordinal","clusters.xyz","particles.tid"])
                self.assertEqual(int(rows["events.original_ordinal"]),index%5)
                np.testing.assert_array_equal(rows["particles.tid"],[42,17])
                self.assertEqual(rows["clusters.xyz"].shape,(5,3))
                self.assertLessEqual(len(source._files),1)
            restored=pickle.loads(pickle.dumps(source))
            self.assertFalse(restored._files)
            np.testing.assert_array_equal(restored.read(2,["particles.tid"])["particles.tid"],[42,17])
            restored.close()
            for value in [-6,5]:
                with self.assertRaises(IndexError): source.read(value,POINT_FIELDS)
            with self.assertRaises(TypeError): source.read(1.2,POINT_FIELDS)
            with self.assertRaises(ValueError): source.read(0,["clusters.offsets"])
        self.assertFalse(source._files)

    def test_identity_uses_ordinal_not_native_event(self):
        with ClusterSource(self.root,"train") as source:
            row=source.identities([4,0,2])[0]
            self.assertEqual(row["original_ordinal"],0)
            self.assertEqual(row["native_event"],1)
            self.assertEqual(source.overlap_key(row),("1"*64,0))
            aliases=source.identities([-1,0,4])
            self.assertEqual(aliases[-1],aliases[4])

    def test_digest_accepts_iterators_and_rechecks_metadata(self):
        with ClusterSource(self.root,"train") as source:
            self.assertEqual(source.content_digest([0,2]),source.content_digest(iter([0,2])))
            (self.root/"geometry/channels.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"metadata/geometry changed"):
                source.content_digest([0,2])

    def test_noise_masks_pid_and_point_order(self):
        with ClusterSource(self.root,"train") as source:
            rows=source.read(0,POINT_FIELDS)
        labels,masks=cluster_projection(rows)
        np.testing.assert_array_equal(labels["seg_target"],[3,3,9,-1,-1])
        np.testing.assert_array_equal(labels["pid_target"],[0,0,1,-100,-100])
        np.testing.assert_array_equal(labels["noise_target"],[0,0,0,1,1])
        native,masks=cluster_projection(rows,noise_policy=NATIVE_NOISE_POLICY)
        np.testing.assert_array_equal(native["noise_target"],[0,0,0,-100,1])
        np.testing.assert_array_equal(masks["noise_target"],[1,1,1,0,1])

    def test_part_metadata_and_offset_corruption(self):
        path=self.root/"train/part-0000.h5"
        with h5py.File(path,"r+") as f: f.attrs["plan_id"]="wrong"
        with ClusterSource(self.root,"train") as source:
            with self.assertRaisesRegex(ValueError,"attributes"): source.read(0,POINT_FIELDS)
        with h5py.File(path,"r+") as f:
            f.attrs["plan_id"]="fixture-plan"
            f["particles/offsets"][:]=[0,4,2]
        with ClusterSource(self.root,"train") as source:
            with self.assertRaisesRegex(ValueError,"monotonic"): source.read(0,["particles.tid"])

    def test_path_escape_and_unsupported_product(self):
        path=self.root/"manifest.json"
        original=json.loads(path.read_text())
        altered=json.loads(path.read_text());altered["splits"]["train"][0]["path"]="../escape.h5"
        path.write_text(json.dumps(altered))
        with self.assertRaises(ValueError): ClusterSource(self.root,"train")
        original["format"]="cvt-strips-h5-v1";path.write_text(json.dumps(original))
        with self.assertRaises(ValueError): ClusterSource(self.root,"train")

    def test_content_drift_with_unchanged_manifest(self):
        with ClusterSource(self.root,"train") as source: before=source.content_digest([0,2])
        with h5py.File(self.root/"train/part-0000.h5","r+") as f: f["clusters/xyz"][0,0]=99
        with ClusterSource(self.root,"train") as source:
            self.assertNotEqual(before,source.content_digest([0,2]))

    def test_spawn_and_fork_workers_use_process_local_handles(self):
        for method in ("spawn", "fork"):
            if method not in multiprocessing.get_all_start_methods():
                continue
            with self.subTest(method=method), ClusterSource(self.root,"train",max_open=1) as source:
                source.read(0,POINT_FIELDS)
                context=multiprocessing.get_context(method)
                queue=context.Queue()
                process=context.Process(target=worker_read,args=(source,queue))
                process.start()
                try:
                    self.assertEqual(queue.get(timeout=20),(4,True,1))
                    process.join(timeout=20)
                    self.assertEqual(process.exitcode,0)
                finally:
                    if process.is_alive():process.terminate();process.join()
                    queue.close();queue.join_thread()
                self.assertEqual(int(source.read(0,["events.original_ordinal"])["events.original_ordinal"]),0)

    def test_zero_length_cluster_domain_and_owner_conflicts(self):
        path=self.root/"train/part-0000.h5"
        with h5py.File(path,"r+") as f:
            group=f["clusters"]
            for key in list(group):
                if key=="offsets":continue
                value=group[key][5:];del group[key];group.create_dataset(key,data=value)
            group["offsets"][:]=[0,0,5]
        with ClusterSource(self.root,"train") as source:
            empty=source.read(0,POINT_FIELDS)
            self.assertEqual(empty["clusters.xyz"].shape,(0,3))
            self.assertEqual(len(cluster_projection(empty)[0]["seg_target"]),0)
            rows=source.read(1,POINT_FIELDS)
        rows["particles.tid"][:]=17
        with self.assertRaisesRegex(ValueError,"association"):cluster_projection(rows)

    def test_missing_field_dtype_and_terminal_offsets(self):
        for kind in ("missing", "dtype", "terminal"):
            with self.subTest(kind=kind):
                root=make_publication(self.root.parent/kind)
                with h5py.File(root/"train/part-0000.h5","r+") as f:
                    if kind=="missing":del f["clusters/primary_pid"]
                    elif kind=="dtype":
                        value=f["clusters/primary_pid"][:].astype(np.float64)
                        del f["clusters/primary_pid"];f.create_dataset("clusters/primary_pid",data=value)
                    else:f["clusters/offsets"][-1]+=1
                with ClusterSource(root,"train") as source:
                    with self.assertRaisesRegex(ValueError,"Invalid clusters"):source.read(0,POINT_FIELDS)

    def test_manifest_and_part_failure_matrix(self):
        for kind in ("incomplete", "duplicate", "wrong_split", "missing_file", "symlink",
                     "part_incomplete", "part_key", "event_count", "offset_dtype", "offset_shape"):
            with self.subTest(kind=kind):
                root=make_publication(self.root.parent/kind)
                manifest_path=root/"manifest.json"
                manifest=json.loads(manifest_path.read_text())
                part=root/"train/part-0000.h5"
                if kind=="incomplete":manifest["status"]="partial"
                elif kind=="duplicate":manifest["splits"]["train"].append(manifest["splits"]["train"][0])
                elif kind=="wrong_split":manifest["splits"]["train"][0]["split"]="test"
                elif kind=="missing_file":part.unlink()
                elif kind=="symlink":
                    outside=root.parent/"outside.h5";outside.write_bytes(part.read_bytes())
                    part.unlink();part.symlink_to(outside)
                else:
                    with h5py.File(part,"r+") as f:
                        if kind=="part_incomplete":f.attrs["complete"]=0
                        elif kind=="part_key":f.attrs["part_key"]="train/wrong.h5"
                        elif kind=="event_count":f.attrs["events"]=3
                        else:
                            del f["clusters/offsets"]
                            f.create_dataset("clusters/offsets",data=np.array(
                                [0,5,10] if kind=="offset_dtype" else [0,10],
                                dtype=np.int32 if kind=="offset_dtype" else np.int64))
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError,FileNotFoundError)):
                    with ClusterSource(root,"train") as source:source.read(0,POINT_FIELDS)
