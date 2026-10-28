# CLAS12 external neural benchmarks

Independent source and CLAS12 adaptations for the neural baselines compared in
FM4NPP. This repository is separate from the CLAS12 foundation-model repository;
it does not import its Mamba models, trainers, or checkpoint code.

## Initial port

| Method | What is here | Current validation |
|---|---|---|
| EggNet | Original upstream source/history; CLAS12 full-event PyG exporter | Export round trip on native-format synthetic RaggedMmap events |
| Exa.TrkX | Original publication pipeline source/history | Source integrity verified; CLAS12 pipeline adaptation pending |
| GATConv, GCNConv, GraphConv, SAGEConv | Our paper-based reconstruction using PyTorch Geometric | CPU forward/backward, parameter update, strict checkpoint reload |
| OneFormer3D | Original source/history including S3DIS configuration | Source integrity verified; CLAS12 semantic-only adaptation pending |

The FM4NPP authors' external-baseline adaptations were not present in the public
`FM4NPP/PP_collision` tree inspected at
`863cb28ad798d4753569737563a754cb2a0e6284`, nor in the locally supplied downstream
archive. These are upstream source imports plus our new integration, **not a
recovered copy of the authors' private benchmark code**. No benchmark convergence,
GPU execution, or reproduction of the published scores is claimed.

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
python -m pip install -e '.[graphs]'
python scripts/verify_sources.py
python -m unittest discover -s tests -v
```

The `graphs` extra is sufficient for our graph classifiers and EggNet data export.
It does not install the three upstream applications. Each has its own dependencies;
use separate environments for those. Original requirement files are preserved in
`vendor/`. The initial CPU checks used PyTorch 2.5.1 and PyG 2.4.0. They do not test
EggNet's GPU neighbor search or Exa.TrkX/OneFormer3D runtimes.

## Use CLAS12 data

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
`pid_target_<split>`/`noise_target_<split>`, `metadata.json`, and
`samples_<split>.jsonl`. It requires complete event samples and checks all selected
point-target alignments. It never silently skips events or truncates their points.
Indices refer to original dataset rows, independently of manifest role names.

The EggNet export uses its upstream `hit_*` feature convention and writes an
`export.json` recording identities, normalization, node-feature names and cohort
fingerprint. These files satisfy the model input convention; an adapted EggNet
training configuration/loss/validation path is still required before training.
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
this repository, normally in sibling `result_deep_storage/benchmarks`. Commit source,
recipes, tests and provenance. A completed run must additionally record this repo's
commit and dirty patch, the source lock, complete environment, data identity,
cohort fingerprint, seed, selection rule and checkpoint hash.

`predictions.save_predictions` checks a complete one-to-one point mapping and
restores original order. Its versioned output is a new interchange format; the
CLAS12 shared evaluator still needs a reader for it. Do not compare upstream native
scores directly to CLAS12 scores: matching, background and cohort definitions must
be shared. The canonical data identity uses source filename plus event number;
datasets must use consistent original event numbering and source-file naming.

## Attribution

Please cite [FM4NPP](https://arxiv.org/abs/2508.14087) and each underlying method.
See [THIRD_PARTY.md](THIRD_PARTY.md) for source links and per-project licenses.
Imported code retains its own license; this repository does not relicense it.
