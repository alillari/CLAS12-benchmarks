# GravNet and GarNet cluster noise benchmark

## Data and split

The cohort comes from the `cvt_alex_new_simulation_20261118_001/cluster`
publication on a02c. It keeps the published HDF5 train, validation, and test
splits and draws complete events uniformly without replacement. The seed-71
cohort uses 12,000 training, 2,400 validation, and 4,000 held-out events. Each
budget takes a fixed prefix of the same training event roster. The cache
records source event indices and original ordinals, publication hashes and
cache hashes. The publication audit checked all 499 shard headers and cluster
offset tables. Each historical comparison uses its own data and truth receipt. An optional FM cohort import accepts a matching publication plan
ID and verifies event identities.

A signal cluster belongs to a direct-primary MC segment with at least four owned
clusters in the complete event. The supervised models use only the measured
cluster center. The four model inputs for the graph arms are
`(eta, cos(phi), sin(phi), transverse radius)`. Training supplies the eta and
radius scales for each label budget. The code retains every measurement,
including background, and keeps events with no measurements in the roster.
The COATJAVA reference flags `reco_segment < 0` on the same first-pass cluster
rows. Detector and layer fields enter the diagnostics after inference.

## Model definitions

GravNet constructs a fresh k-nearest-neighbor graph per layer in learned
coordinates. Each neighbor message has weight `exp(-d²)`. The layer reduces
messages by mean and max and concatenates the result with the input before
a tanh projection. Native PyTorch computes graph membership in bounded query
blocks over each event. It recomputes the selected distances with autograd.
Distances tied at the cutoff are ordered by the original observed features.
Saved message tensors scale with the retained neighbors. Distance evaluation
is quadratic in occupancy, with a stable sort of the candidates for each
query. The event size remains unrestricted.
[Original paper, section 3](https://arxiv.org/html/1902.07987#S3)

The PyG operator uses compiled nearest-neighbor search and scales squared
distances by 10 before the exponential. The native implementation starts at
the paper's distance scale of 1 and exposes that scale in its configuration.
[PyG operator source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gravnet_conv.html)

GarNet learns signed distances from each measurement to its shared aggregators.
It uses `exp(-abs(d))` on the way to and from the aggregators, with mean and
max reductions across real measurements. The layer follows the paper's
message definition. The authors' TensorFlow implementation additionally
concatenates affinity features. The benchmark uses a common residual,
per-measurement LayerNorm and multilayer readout for both graph families.
[Authors’ reference implementation](https://github.com/jkiesele/caloGraphNN/blob/master/caloGraphNN.py)

Padded points never contribute to messages or loss. GravNet and GarNet run
without PyG, `torch_cluster`, or `pyg-lib`. SAGEConv uses the existing graph
classifier and static geometry edges with batch-block sparse CSR adjacency.
The CUDA test compares that path with the existing edge-list implementation
and checks its backward pass. The runtime sets full float32 matmul precision
after observing shape-dependent rounding at the GB10 image's default.

The FM4NPP controls use the native noise attention head, fixed FM4NPP
cylindrical normalization, and the calibrated12 unit-v2 Hilbert ordering.
The frozen control uses the final normalized state of the SHA256-pinned
Campaign 5 M4 backbone. It trains only the attention head. The frozen cache
holds float32 embeddings of measured event coordinates in the original row
order. The extractor checks that the cached and online heads agree on an
initial batch from each split. This is a final-state readout. The historical
noise trainer also supports a learned mixture of backbone layers.

Ronayette's published input list describes strip endpoints, strip ID and
layer ID. This publication has first-pass cluster centers, cluster endpoints,
a seed-strip field and detector addresses. A cluster can cover several strips.
No per-strip comparison was included on this release.
[Ronayette, GDR QCD presentation, slide 7](https://indico.in2p3.fr/event/40045/contributions/178017/attachments/104070/162367/gdr_qcd_mathieu_ronayette.pdf)

## Replaying the study

From the repository root, install the full study dependencies with
`python -m pip install -e '.[graphs,hdf5,evaluation,plotting]'` in an
environment with the appropriate PyTorch build. GravNet and GarNet also run
with `tracking,hdf5,plotting` when the configuration excludes SAGEConv and
the optional native FM controls. The GB10 run used the local ARM64 image
`clas12-explore/conv-noise-pyg24-a02c-20261111-v1:local`.

Set `DATA_ROOT` to the cluster publication root containing `manifest.json` and
`RUN_ROOT` to a writable artifact directory. Set `CLAS12_FM_SOURCE` to the pinned native source
checkout for the FM controls. On a02c it came from the immutable source snapshot
for `conv-m4-c-10k-s43-20261126-v1`, based on CLAS12 commit
`692aa26ee57bdb4d55151568009fbf26e3d8aef8`. Set `GRAPH_NOISE_CONFIG` if the
checkpoint is mounted at a location different from the one in the supplied
configuration.

```bash
scripts/run_graph_noise_study.sh cohort "$DATA_ROOT" "$RUN_ROOT/cohort"
scripts/run_graph_noise_study.sh pilot "$RUN_ROOT/cohort" "$RUN_ROOT/pilot" cpu
scripts/run_graph_noise_study.sh search "$RUN_ROOT/cohort" "$RUN_ROOT/search" cpu
scripts/run_graph_noise_study.sh final "$RUN_ROOT/cohort" "$RUN_ROOT/final" cpu \
    "$RUN_ROOT/search/summary.json"
scripts/run_graph_noise_study.sh plot "$RUN_ROOT/final" "$RUN_ROOT/plots"
```

For the three graph models with no native FM4NPP dependencies, leave
`CLAS12_FM_SOURCE` unset, use the same cohort, pilot and search commands, and
skip the FM controls in the held-out fit:

```bash
scripts/run_graph_noise_study.sh final "$RUN_ROOT/cohort" "$RUN_ROOT/graph-only" cpu \
    "$RUN_ROOT/search/summary.json" --without-fm
scripts/run_graph_noise_study.sh plot "$RUN_ROOT/graph-only" "$RUN_ROOT/graph-only-plots"
```

For a quick test on a smaller publication, the cohort command accepts
`--train`, `--validation`, `--test` and `--seed`. Point `GRAPH_NOISE_CONFIG`
at a copy of the recipe with the phase budgets reduced to fit that cohort.
The CPU workflow regression test uses a synthetic HDF5 v2 publication:

```bash
python -m unittest discover -s tests -p test_graph_noise_workflow.py -v
```

The optional frozen-feature cache runs on CUDA. Use the checkpoint and SHA256
recorded in the configuration. Then run either budget through the same
validation-selected model recipes. The environment's native FM4NPP imports must
be on `PYTHONPATH` for direct Python invocations.

```bash
export PYTHONPATH="$PWD/src:$CLAS12_FM_SOURCE"
python -m clas12_benchmarks.graph_noise_fm_cache \
    --cohort "$RUN_ROOT/cohort" --output "$RUN_ROOT/m4-features" \
    --checkpoint "$M4_CHECKPOINT" --sha256 "$M4_SHA256"
python -m clas12_benchmarks.graph_noise \
    --cache "$RUN_ROOT/cohort" --config configs/graph_noise_cluster_v2.json \
    --phase low_label --output "$RUN_ROOT/low_label" --device cpu \
    --selection "$RUN_ROOT/search/summary.json" --fm-features "$RUN_ROOT/m4-features"
```

Use `--phase mid_label` with a fresh directory for the 4,096-event budget.
`scripts/profile_graph_noise.py` reports a separate GB10 profile from saved
checkpoints and uses validation events. EdgeExpert jobs take the cohort,
native source and prior results as read-only mounts and write into a new
managed results directory.

## Selection and metrics

The pilot verifies the pipeline. The search compares nine short validation-only
fits over architecture, width, depth, propagated features, neighbor and
aggregator counts, learning rate and class balance. Each candidate selects an
update by its pooled validation cross-entropy. The selected update's mean
validation noise rejection at 95%, 98%, 99%, and 99.5% signal retention selects
one configuration per architecture. Full and label-budget jobs refit those
configurations. Their validation-calibrated thresholds apply unchanged to the
held-out scores. The evaluator opens the held-out archive after the fits have
selected their checkpoints. Frozen features are computed earlier from measured
coordinates using the fixed pretrained backbone.

Metrics pool clusters across events. Signal retention is signal recall and
noise rejection is noise recall. Each score archive preserves event offsets
and measurement alignment. The threshold uses strict `P(noise) > threshold`.
The saved curves use all distinct probabilities. Confidence intervals
resample complete paired events at fixed thresholds. They quantify test-cohort
sampling variation for the fitted checkpoints. Timing reports specify the
measured device, workload and whether frozen features were precomputed.
Each run records resolved settings, a provenance receipt, the cohort fingerprint,
the validation selection rule and the selected checkpoint hashes. Store these
reports and the binary artifacts together outside version control.
