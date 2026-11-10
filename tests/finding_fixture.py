"""Full events containing a four-cluster target, excluded MC and pure overlay."""
import json

import h5py
import numpy as np

from hdf5_fixture import make_publication


def make_finding_publication(root, provenance=True):
    root = make_publication(root)
    schema = json.loads((root / "schema.json").read_text())
    values = {"xyz": np.column_stack((np.arange(9)+1, np.ones(9), np.arange(9))).astype("f4"),
              "primary_segment": np.array([3]*4+[9]*3+[-1]*2, dtype="i4"),
              "primary_tid": np.array([17]*4+[42]*3+[-1]*2, dtype="i4"),
              "reco_segment": np.array([0]*4+[1]*3+[-1]*2, dtype="i4"),
              "primary_pid": np.full(9, 211, dtype="i4"), "primary_pid_valid": np.ones(9, bool),
              "primary_noise": np.array([0]*7+[1]*2, dtype="i1"), "primary_noise_valid": np.ones(9, bool)}
    if provenance:
        values.update(background_only=np.array([False]*8+[True]),
                      ownership=np.array([1]*7+[3,0], dtype="i1"),
                      origin_noise=np.array([False]*8+[True]),
                      purity=np.array([1.]*7+[0.,0.], dtype="f4"))
    schema["fields"]["clusters"] = {k: {"dtype": v.dtype.str, "row_shape": list(v.shape[1:])}
                                        for k, v in values.items()}
    for path in root.rglob("*.h5"):
        with h5py.File(path, "r+") as file:
            count = int(file.attrs["events"])
            del file["clusters"]
            for name, value in values.items():
                file.create_dataset("clusters/"+name, data=np.concatenate([value]*count))
            file.create_dataset("clusters/offsets", data=np.arange(count+1, dtype="i8")*9)
    (root / "schema.json").write_text(json.dumps(schema))
    return root
