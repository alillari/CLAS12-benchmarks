from __future__ import annotations
import argparse
import json
from pathlib import Path

from .data import Cohort, eggnet_features, positive_edges


def export_eggnet(cohort, output, max_true_edges):
    import numpy as np
    import torch
    from torch_geometric.data import Data

    output = Path(output)
    # Never mix a new cohort with old event files.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for role, members in cohort.roles.items():
        (output / role).mkdir()
        for number in range(len(members)):
            event = cohort.event(role, number)
            features = eggnet_features(event)
            data = Data(num_nodes=len(features))
            for column in range(12):
                data[f"hit_feature_{column}"] = torch.from_numpy(features[:, column].copy())
            data.hit_id = torch.from_numpy(event.original_point_index.copy())
            # EggNet reserves particle id zero for background; CLAS12 uses -1.
            labels = event.labels["seg_target"]
            data.hit_particle_id = torch.from_numpy(np.where(labels < 0, 0, labels + 1))
            data.track_edges = torch.from_numpy(positive_edges(labels, max_true_edges))
            data.event_id = [str(number)]
            data.clas12_identity_json = json.dumps(event.identity, sort_keys=True)
            data.clas12_cohort_sha256 = cohort.sha256
            filename = f"{role}/event{number:09d}.pyg"
            torch.save(data, output / filename)
            records.append({"file": filename, "identity": event.identity, "n_points": len(features)})
    report = {"schema": "clas12_eggnet_export_v1", "cohort_sha256": cohort.sha256,
              "node_features": [f"hit_feature_{i}" for i in range(12)],
              "normalization": {name: {key: spec[key] for key in ("eta_scale", "radius_scale")}
                                for name, (spec, _, _) in cohort.datasets.items()},
              "events": records}
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Independent CLAS12 baseline adapters")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate all selected events and identities")
    inspect.add_argument("manifest", type=Path)
    export = sub.add_parser("export-eggnet", help="Export full events to upstream EggNet PyG files")
    export.add_argument("manifest", type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--max-true-edges", type=int, default=1_000_000)
    args = parser.parse_args()
    cohort = Cohort(args.manifest)
    if args.command == "inspect":
        report = {"cohort_sha256": cohort.sha256, "roles": {}}
        for role, members in cohort.roles.items():
            points = sum(len(cohort.event(role, i).raw_features) for i in range(len(members)))
            report["roles"][role] = {"events": len(members), "points": points}
    else:
        report = export_eggnet(cohort, args.output, args.max_true_edges)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
