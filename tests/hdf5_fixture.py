"""Small independently authored publication: IDs deliberately differ from rows."""
import json
from pathlib import Path

import h5py
import numpy as np


def make_publication(root, *, cluster_fields=None):
    root = Path(root)
    root.mkdir()
    (root / "geometry").mkdir()
    (root / "geometry/channels.json").write_text('{"unit":"cm"}')
    (root / "recipe.json").write_text('{"reference":"observed_primary_first_layer_hilbert"}')
    manifest = {"format": "cvt-clusters-h5-v1", "status": "complete", "plan_id": "fixture-plan",
                "source_manifest_id": "fixture-inventory", "sources": [], "splits": {}}
    fields = {}
    for sid, split in enumerate(("train", "validation", "test")):
        manifest["sources"].append({"id": sid, "split": split, "count": 100,
                                     "signal": {"sha256": str(sid + 1) * 64}})
        (root / split).mkdir()
        manifest["splits"][split] = []
        ordinal = 0
        for part, count in enumerate((2, 1, 2)):
            name = f"{split}/part-{part:04d}.h5"
            manifest["splits"][split].append({"path": name, "split": split, "events": count})
            with h5py.File(root / name, "w") as f:
                f.attrs.update(format=manifest["format"], complete=1, plan_id="fixture-plan",
                               part_key=name, events=count)
                domains = {"events": {
                    "source_id": np.full(count, sid, np.int32),
                    "original_ordinal": np.arange(ordinal, ordinal + count, dtype=np.int64),
                    "run": np.full(count, 11, np.int32),
                    "event": np.arange(ordinal + 1, ordinal + count + 1, dtype=np.int32)}}
                one = {
                    "clusters": {
                        "xyz": np.array([[1,0,1],[0,2,2],[-3,0,3],[0,-4,4],[5,0,5]], np.float32),
                        "primary_segment": np.array([3,3,9,-1,-1], np.int32),
                        "reco_segment": np.array([0,0,1,-1,-1], np.int32),
                        "primary_tid": np.array([17,17,42,-1,-1], np.int32),
                        "primary_pid": np.array([211,211,-321,0,0], np.int32),
                        "primary_pid_valid": np.array([1,1,1,0,0], bool),
                        "primary_noise": np.array([0,0,0,-1,1], np.int8),
                        "primary_noise_valid": np.array([1,1,1,0,1], bool)},
                    "particles": {
                        "tid": np.array([42,17], np.int32),
                        "observed_momentum": np.array([[1,0,0],[.3,-.4,1.2]], np.float32),
                        "observed_valid": np.ones((2,3), bool)},
                    "reco/cvt_tracks": {
                        "native": np.array([(100,),(200,)], dtype=[("ID","<i2")]),
                        "benchmark_p": np.array([1.3,1], np.float32),
                        "benchmark_theta": np.array([np.arctan2(.5,1.2),np.pi/2],np.float32),
                        "entrance_phi": np.array([np.arctan2(-.4,.3),0],np.float32),
                        "entrance_row": np.array([0,1],np.int32),
                        "purity_tid": np.array([17,42],np.int32),
                        "purity_valid": np.ones(2,bool), "unique_owner": np.ones(2,bool),
                        "purity": np.ones(2,np.float32)},
                    "reco/cvt_trajectory": {
                        "native": np.array([(100,.6,-.8),(200,1.,0.)],
                                           dtype=[("id","<i2"),("cx","<f4"),("cy","<f4")])},
                    "reco/contributors": {
                        "pass_id": np.array([0,0,1,1],np.int32),
                        "track_row": np.array([0,1,0,1],np.int32),
                        "tid": np.array([17,42,17,42],np.int32),
                        "count": np.array([2,1,2,1],np.int32)}}
                if cluster_fields is not None:
                    one["clusters"] = cluster_fields
                one["reco/cvtrec_tracks"] = {key: value.copy() for key,value in one["reco/cvt_tracks"].items()}
                one["reco/cvtrec_tracks"]["native"]["ID"] = [700,900]
                one["reco/cvtrec_tracks"]["benchmark_p"] *= 2
                one["reco/cvtrec_tracks"]["entrance_phi"][:] = [1.2,-.5]
                one["reco/cvtrec_trajectory"] = {"native": np.array(
                    [(700,np.cos(1.2),np.sin(1.2)),(900,np.cos(-.5),np.sin(-.5))],
                    dtype=[("id","<i2"),("cx","<f4"),("cy","<f4")])}
                for domain, columns in one.items():
                    domains[domain] = {key: np.concatenate([array] * count) for key,array in columns.items()}
                    f.create_dataset(domain + "/offsets", data=np.arange(count + 1,dtype=np.int64)*len(next(iter(columns.values()))))
                for domain, columns in domains.items():
                    fields[domain] = {}
                    for key,array in columns.items():
                        f.create_dataset(domain + "/" + key,data=array)
                        fields[domain][key] = {"dtype":array.dtype.descr if array.dtype.names else array.dtype.str,
                                               "row_shape":list(array.shape[1:])}
            ordinal += count
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "schema.json").write_text(json.dumps({"format":manifest["format"],"fields":fields}))
    return root
