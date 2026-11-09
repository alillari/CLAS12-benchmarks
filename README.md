# CLAS12 external neural benchmarks

Independent source and CLAS12 adaptations for external neural benchmarks. The
initial tracking and semantic ports follow the FM4NPP comparisons. The
[momentum guide](docs/MOMENTUM.md) covers the shared track manifest and evaluator.
The [implementation plan](docs/MOMENTUM_PLAN.md) describes the Miller track
fitter and the Jahin CMS trigger GNN. The [HDF5 support plan](docs/DATA_FORMAT_PLAN.md)
describes native cluster access while preserving legacy data workflows.
The [testing plan](docs/TESTING_PLAN.md) defines regression checks and acceptance
gates for existing workflows and the HDF5 implementation.
This repository is separate from the
CLAS12 foundation-model repository and does not import its models, trainers or checkpoint code.

## Initial port

| Method | Included support |
|---|---|
| EggNet | Upstream source, CLAS12 exporter, training and DBSCAN evaluation |
| Exa.TrkX | Upstream embedding/filter forwards and Torch Interaction Network port |
| GATConv, GCNConv, GraphConv, SAGEConv | Paper-based semantic graph reconstruction using PyTorch Geometric |
| OneFormer3D | Upstream source and S3DIS configuration; CLAS12 adaptation pending |

The FM4NPP authors' external-baseline adaptations were not present in the public
`FM4NPP/PP_collision` tree inspected at
`863cb28ad798d4753569737563a754cb2a0e6284`, nor in the locally supplied downstream
archive. These are upstream source imports plus our new integration, **not a
recovered copy of the authors' private benchmark code**. See [tracking
usage](docs/TRACKING.md) for cohort preparation, training, shared CLAS12 evaluation and
prediction auditing. Frozen comparisons and corrected-loss recipes are also supported.
Nested label sweeps include metric intervals and learning-curve plotting.

`sources.lock.json` pins imported upstream revisions. Imports use Git subtrees
without squashing, retaining original commits and licenses. The upstream revision
is a starting point selected for this port; FM4NPP's exact external-code revisions
were not specified. `configs/fm4npp_reference.json` records paper settings separately
from executable CLAS12 settings. See [porting details](docs/PORTING.md).

## Layout

- `vendor/`: upstream implementations; make any future modifications explicit.
- `src/clas12_benchmarks/`: our dataset adapters, graph reconstructions, and outputs.
- `configs/`: reference specifications and a cohort-manifest example.
- `tests/`: native-format synthetic data and graph-execution checks.
- `scripts/verify_sources.py`: verify imported files against pinned Git objects.

## Install and verify

Use a dedicated environment with a suitable PyTorch installation:

```bash
python -m pip install -e '.[graphs,tracking,plotting,hdf5]'
python scripts/verify_sources.py
python -m unittest discover -s tests -v
```

The `graphs` extra is sufficient for our graph classifiers and EggNet data export.
It does not install the three upstream applications. Each has its own dependencies;
use separate environments for those. Original requirement files are preserved in
`vendor/`. The standalone tracking runner uses the `tracking` extra and native
Torch graph operations; it runs without those full application environments.
The tracking usage guide describes the optional TensorFlow/Sonnet port-parity check.

## Use CLAS12 data

Native cluster HDF5 (`cvt-clusters-h5-v1`) and legacy RaggedMmap are supported.
Use `--reader hdf5` when preparing the new product; the default reader preserves
the old workflow. See the [HDF5 workflow](docs/TRACKING.md#native-cluster-hdf5)
and [HDF5 momentum packing](docs/MOMENTUM.md#cluster-hdf5-momentum).
HDF5 cohorts reference read-only shards and fingerprint selected content; they
do not require a RaggedMmap conversion. The `hdf5` extra is optional for legacy
loading and scoring. Tracking/upstream exports use the complete source checkout
and the documented editable installation, including `vendor/` and `configs/`.

Start with `configs/cohort.example.json`. Replace the dataset path, normalization
constants, fingerprints, and explicit event lists with those from the actual
CLAS12 comparison. The example is a template, not a scientifically selected split.
Compute the two fingerprints with `sha256sum metadata.json samples_pretrain.jsonl`
inside the data directory. They identify metadata and sample provenance, not the
contents of every large RaggedMmap data file; archive or checksum data separately.

```bash
clas12-benchmarks inspect /absolute/path/cohort.json
clas12-benchmarks export-eggnet /absolute/path/cohort.json \
  --output /absolute/path/result_deep_storage/benchmarks/eggnet/input
```

The reader accepts native `features_<split>`, `seg_target_<split>`, optional
`pid_target_<split>`/`noise_target_<split>`/`coatjava_seg_pred_<split>`, `metadata.json`, and
`samples_<split>.jsonl`. It requires complete event samples and checks all selected
point-target alignments. It never silently skips events or truncates their points.
Indices refer to original dataset rows, independently of manifest role names.

The EggNet export uses its upstream `hit_*` feature convention and writes an
`export.json` recording identities, normalization, node-feature names and cohort
fingerprint. These files satisfy the model input convention and preserve truth
for downstream training/evaluation.
The standalone runner documented in [tracking usage](docs/TRACKING.md)
now supplies a training and evaluation path independently of these exports.
Exports contain truth for supervised training and evaluation: only the twelve
named coordinate features may be passed to model inference.

Coordinate features are `(eta / eta_scale, cos(phi), sin(phi), r / radius_scale)`;
`r` is transverse radius. Normalization must come from the common benchmark
protocol and training data, never fit on test data. This continuous-radius CLAS12
mapping differs from FM4NPP's 48-layer TPC radius encoding. No TPC geometry or dummy
energy is silently applied to CLAS12. For xyz inputs, `node_features` requires an
explicit position-only opt-in; energy-bearing inputs must declare `exyz`.

Graph classifiers return per-point semantic logits. They are PID/noise baselines,
not published momentum-regression models or track-instance classifiers. Class
mapping and ignored-label policy must be set in a task-specific training runner.

## Experiment boundary

Keep datasets, prepared graphs, checkpoints, predictions, logs and plots outside
this repository, normally in sibling `result_deep_storage/benchmarks`. Commit reusable source,
recipes and tests. Keep run-specific provenance and result reports with the
external artifacts. A completed run must additionally record this repo's
commit and dirty patch, the source lock, complete environment, data identity,
cohort fingerprint, seed, selection rule and checkpoint hash.

`predictions.save_predictions` checks a complete one-to-one point mapping and
restores original order. Its versioned output is a new interchange format; the
standalone tracking runner evaluates predictions through the shared CLAS12 metric
module before saving them. The original CLAS12 evaluation CLI still needs a reader
for saved interchange files. Do not compare upstream native
scores directly to CLAS12 scores: matching, background and cohort definitions must
be shared. The canonical data identity uses source filename plus event number;
datasets must use consistent original event numbering and source-file naming.

## Attribution

Please cite [FM4NPP](https://arxiv.org/abs/2508.14087) and each underlying method.
See [THIRD_PARTY.md](THIRD_PARTY.md) for source links and per-project licenses.
Imported code retains its own license; this repository does not relicense it.
