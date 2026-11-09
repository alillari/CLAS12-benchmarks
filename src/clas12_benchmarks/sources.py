"""Legacy event source; arrays and identities remain in their original format."""
import json


def open_ragged(root, field, split):
    """mmap_ninja creates absent directories even for a nominal read-only open."""
    from mmap_ninja import RaggedMmap
    path = root / f"{field}_{split}"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return RaggedMmap(path, mode="r")


class RaggedSource:
    def __init__(self, root, split, spec, requested):
        self.rows, total = {}, 0
        with (root / f"samples_{split}.jsonl").open() as stream:
            for index, line in enumerate(stream):
                if index in requested:
                    self.rows[index] = json.loads(line)
                total += 1
        if set(self.rows) != requested:
            raise ValueError("Invalid event index in legacy dataset")
        self.arrays = {}
        for field in ["features", "seg_target", *spec.get("extra_targets", [])]:
            if field in self.arrays or field not in {
                    "features", "seg_target", "pid_target", "noise_target", "coatjava_seg_pred"}:
                raise ValueError(f"Duplicate or unsupported target: {field}")
            self.arrays[field] = open_ragged(root, field, split)
            if len(self.arrays[field]) != total:
                raise ValueError(f"Event count mismatch for {field}")
        self.count = total

    def __len__(self):
        return self.count

    def read(self, index, fields):
        return {name: self.arrays[name][index] for name in fields}

    def identities(self, indices):
        return {index: self.rows[index] for index in indices}

    def close(self):
        # mmap_ninja owns the read-only NumPy mappings; no HDF5 resources exist.
        pass
