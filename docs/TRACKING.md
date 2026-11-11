# Running and examining tracking experiments

Install a suitable PyTorch build and the tracking dependencies:

```bash
python -m pip install -e '.[tracking]'
```

Keep prepared data and every generated artifact outside the checkout. The
commands below use an external `RUN_ROOT`; replace all paths with your own.
Commit reusable source, tests and configuration recipes. Run-specific manifests,
metrics, provenance, checkpoints, logs, plots and experiment reports belong with
the external run artifacts, not in Git.

## Prepare an event cohort

The native input directory must contain `metadata.json`, `samples_<split>.jsonl`
and row-aligned RaggedMmap directories `features_<split>`, `seg_target_<split>`
and `coatjava_seg_pred_<split>` for the `pretrain` and `test` splits.
The preparation tool expects xyz coordinates and preserves complete events.

```bash
export RUN_ROOT=/absolute/path/result_deep_storage/benchmarks/example
python -m clas12_benchmarks.prepare_cohort /absolute/path/native_data \
  "$RUN_ROOT/cohort" --train 10000 --val 5000 --test 20000 --seed 1337
```

Train and validation are sampled without replacement from `pretrain`; evaluation
is sampled from `test`. The reader rejects duplicate source-file/event identities
across roles. Source files can overlap across roles: this is an event holdout.
Normalization is fitted on training coordinates only. Preparation writes a cohort
manifest, original identities, copied arrays and a population summary externally.

The legacy preparation command above remains unchanged. Native HDF5 preparation
uses the following explicit reader selection.

## Native cluster HDF5

Install the `hdf5` extra in addition to the tracking dependencies. Prepare a
frozen cohort on a machine where the complete product is mounted:

```bash
python -m pip install -e '.[tracking,graphs,hdf5,plotting]'
python -m clas12_benchmarks.prepare_cohort /absolute/path/cvt_cluster_hdf5_20261103_002 \
  "$RUN_ROOT/cohort" --reader hdf5 --train 10000 --val 1000 --test 2000 --seed 1337
clas12-benchmarks inspect "$RUN_ROOT/cohort/cohort.json"
```

Training, validation and evaluation come from the published `train`,
`validation` and `test` splits respectively. No shard payload is copied. The
v2 cohort records metadata/geometry fingerprints, selected-content hashes,
explicit source rows, training-only normalization, and feature/label policies.
Use the existing tracking commands below with its `cohort.json`.

Inputs remain position-only: first-pass cluster xyz in cm is transformed to the
same normalized eta, cosine/sine phi, and radius features. Every cluster is
retained in original order; -1 remains background. `clusters.reco_segment`
supplies the stored COATJAVA grouping. PID maps absolute PDG values to
pion/kaon/proton/electron/other, with -100 for invalid PID. Raw signed PDG,
native noise and validity fields remain available through the source reader.
`Event.masks` preserves supervised validity.

Generic HDF5 cohort preparation defaults to `primary_unassigned_is_noise_v1`,
the historical projection treating unassigned clusters (including contested
ownership) as noise. The [dedicated noise runner](NOISE_TAGGING.md) imports FM
cohorts with the four-cluster rule. Use `--noise-policy primary_valid_only_v1` to
instead supervise only valid native noise labels; invalid entries then use -100
with a false mask. These are different declared label policies.

Physical HDF5 event identity is `(source_manifest_id, source_id,
original_ordinal)`. Native run/event numbers are separate diagnostics. V2
prediction files and audits use this identity; old v1 prediction files retain
their existing source-file/event keys. Training and frozen evaluation verify
consumed source content, including for resume. HDF5 events and derived graphs
use bounded caches; `event_cache_size` in the training JSON controls event
caching (default eight). Existing graph/pair limits fail explicitly and never
truncate a large event.

Legacy manifests with external array roots also fingerprint their selected
payloads for resume and audit. Archived materialized legacy cohorts retain their
existing fingerprint convention. HDF5 extension/sweep preparation accepts a
relocated copy only when the parent's selected bytes still agree.

`--extend-from` preserves training and old validation membership and draws fresh
test events excluding the parent's test set. The existing label-sweep module
also accepts HDF5 parents and writes nested reference cohorts:

```bash
python -m clas12_benchmarks.prepare_label_sweep /absolute/path/cvt_cluster_hdf5_20261103_002 \
  "$RUN_ROOT/cohort" "$RUN_ROOT/sweep"
```

The default sweep requires its anchor of 10,000 training events; use the Python
`prepare(..., budgets=...)` API for custom budget lists. Use a new output directory
for each preparation. A failed preparation does not publish a completed cohort.

## Train and evaluate a frozen cohort

```bash
python -m clas12_benchmarks.tracking --method eggnet \
  --cohort "$RUN_ROOT/cohort/cohort.json" \
  --config configs/tracking.json \
  --metrics /absolute/path/CLAS12/train/downstream/track_finding_metrics.py \
  --output "$RUN_ROOT/eggnet"
```

Use `--method exatrkx` for embedding, filtering and GNN training. Use `--device
cpu` for a CPU run. `--smoke` performs a bounded training/validation check without
evaluating the test set. `--resume` restores the last checkpoint and rejects
incompatible source, recipe, cohort, evaluator or prepared-array fingerprints.
Use a fresh output directory for an independent experiment.

The runner uses pinned upstream EggNet, Exa.TrkX embedding and filter forwards.
The Interaction Network is a Torch port of the pinned TensorFlow/Sonnet model.
The inputs are position-only adaptations: normalized eta, cos(phi), sin(phi) and
continuous radius. EggNet concatenates each point with two geometry neighbors.
Neither model receives truth or COATJAVA predictions as input. Background points
are never supervised as one common track. Recipe files specify architecture,
training and validation-only inference-cut grids; they are starting configurations,
not guarantees of convergence or reproductions of published performance.

Exa.TrkX trains its stages sequentially, so equal updates per stage do not match
total compute between architectures. Checkpoint and cut selection use validation;
test predictions are produced after checkpoint hashes and cuts are frozen.

## Inspect and audit outputs

`history.jsonl` records training/validation progress. `selected_checkpoints.json`
records selected updates and checkpoint hashes; `frozen_selection.json` records
inference cuts. `provenance.json` and `pip_freeze.txt` record source, environment,
recipe, cohort and evaluator identities. Preserve these with the run.

`summary.json` contains model and stored COATJAVA metrics on the same points.
`per_event_metrics.jsonl` supports population diagnostics, and `predictions/`
contains point-aligned labels. The shared evaluator reports signal ARI, global
track purity (accepted matches / predicted tracks) and global track efficiency
(accepted matches / truth tracks), as well as split, merge and background metrics.
Global count ratios and means over events are distinct aggregations. Inspect
the snapshotted evaluator for matching thresholds and undefined-value conventions.

```bash
python scripts/audit_tracking_predictions.py --run "$RUN_ROOT/eggnet" \
  --cohort "$RUN_ROOT/cohort/cohort.json" \
  --output "$RUN_ROOT/eggnet_audit.json"
```

The audit reloads every prediction, verifies source/event/point identities,
recomputes summaries and checks selected checkpoint hashes. An artifact-presence
or process-exit check alone does not establish valid scientific results.

## Development checks

```bash
python -m unittest discover -s tests -v
python scripts/check_tracking_pipeline.py --output "$RUN_ROOT/synthetic_check" \
  --metrics /absolute/path/CLAS12/train/downstream/track_finding_metrics.py
python scripts/verify_sources.py
```

`scripts/check_exatrkx_tf_port.py` checks forward and gradient parity with the
original TensorFlow/Sonnet Interaction Network. Its optional dependencies are
documented in the script; they are not required for Torch training.

## Corrected-loss recipe and frozen comparisons

`configs/tracking.json` enables the upstream EggNet Euclidean
squared-hinge objective, warmup/cosine learning rate, and tracking-score checkpoint
selection for EggNet/GNN. The compatibility recipe `configs/first_real_runs.json`
retains the earlier class-balanced pair objective. These are explicit choices;
check the objective and selection rule when comparing runs. The dated recipes
explicitly retain `legacy_index` edge direction for compatibility. New-run
defaults use `geometry`: order filter endpoints by 3D distance from the coordinate origin, then xyz
for ties, independently of point row order. Saved configurations without an
orientation retain legacy direction during frozen inference. Retrain with the new
recipe before interpreting any performance difference; changing the direction of
inputs to an existing asymmetric filter changes its predictions.

Expand a prepared cohort while preserving training members/order and normalization:

```bash
python -m clas12_benchmarks.prepare_cohort /absolute/path/native_data \
  "$RUN_ROOT/expanded" --train 10000 --val 5000 --test 20000 \
  --extend-from /absolute/path/previous/cohort
```

Existing validation members are retained; new evaluation members exclude the
previous evaluation set. An event holdout still permits shared source files.

```bash
python scripts/run_tuned_comparison.py --method eggnet \
  --cohort "$RUN_ROOT/expanded/cohort.json" \
  --original-cohort /absolute/path/previous/cohort/cohort.json \
  --parent /absolute/path/previous/eggnet \
  --config configs/tracking.json \
  --metrics /absolute/path/CLAS12/train/downstream/track_finding_metrics.py \
  --output "$RUN_ROOT/comparison"
```

This trains the configured pipeline, evaluates the parent pipeline with frozen
weights and cuts, audits both, and writes `comparison.json` with paired event
bootstrap intervals. Parent training identities, normalized inputs, truth,
architecture sources and checkpoint hashes must remain compatible. Frozen
inference can also be invoked through `python -m clas12_benchmarks.evaluate_frozen`.
All comparison outputs stay under the external output directory.

## Nested training-label sweeps

Prepare nested budgets around an existing cohort with the parent training size
included. The default budgets are 100, 1,000, 10,000 and 100,000 training events:

```bash
python -m clas12_benchmarks.prepare_label_sweep /absolute/path/native_data \
  /absolute/path/previous/cohort "$RUN_ROOT/sweep_cohorts" --seed 1337
python scripts/run_label_sweep.py --method eggnet \
  --cohorts "$RUN_ROOT/sweep_cohorts" \
  --config configs/label_budget_sweep.json \
  --metrics /absolute/path/CLAS12/train/downstream/track_finding_metrics.py \
  --output "$RUN_ROOT/eggnet_sweep"
```

Run the second command with `--method exatrkx` and a separate output directory
for the other pipeline. Arms run sequentially to avoid competing for the GPU.
The runner's `--budgets` option can select prepared manifests by training size.
Preparation preserves fixed validation/evaluation identities and order, nests
training members and fits normalization separately on each training subset.
Budgets count fully labeled training events; fixed validation labels are additional.

The supplied recipe fixes optimizer updates per stage across budgets. Consequently,
the number of passes through the data decreases as the label budget grows. This
controls updates within each architecture, not training to convergence or compute
across architectures. Selected checkpoints can have seen fewer distinct events
than the available budget; actual exposure is included in the report.

The runner writes `sweep_progress.json`, `sweep_partial.json` and
`sweep_summary.json`; each arm contains `run/` and `audit.json`. Summaries include
95% event-bootstrap intervals. These intervals describe evaluation sampling
uncertainty conditional on trained weights, not seed or training-subset variance.

```bash
python scripts/plot_label_sweep.py \
  --eggnet "$RUN_ROOT/eggnet_sweep/sweep_summary.json" \
  --exatrkx "$RUN_ROOT/exatrkx_sweep/sweep_summary.json" \
  --output "$RUN_ROOT/label_sweep.png"
```

The plot shows signal ARI, global track purity and global track efficiency,
conditional bootstrap bands and stored COATJAVA reference values. Install
Matplotlib separately for plotting. Keep plots and their input summaries with the
external run artifacts.

## Diagnose frozen Exa.TrkX graph construction

```bash
python -m clas12_benchmarks.exatrkx_diagnostics --run "$RUN_ROOT/exatrkx" \
  --cohort "$RUN_ROOT/cohort/cohort.json" --split val --device cpu \
  --output "$RUN_ROOT/exatrkx_diagnostics"
python scripts/plot_exatrkx_diagnostics.py --input "$RUN_ROOT/exatrkx_diagnostics"
```

The diagnostic verifies the saved cohort, data arrays, checkpoint hashes, model
sources and evaluator. It uses every event in the requested role by default;
`--max-events` provides a bounded smoke check. No checkpoint or cut is retrained
or recalibrated. Use a fresh external output directory. Plotting needs Matplotlib.

Data verification uses the trainer's cohort fingerprint contract for materialized
legacy arrays, externally referenced legacy arrays and native cluster HDF5.
Inference and active reader sources must match the saved run; use its matching
source snapshot when inspecting older checkpoints. Missing source hashes are
rejected. Undefined metrics (for example, purity with no reconstructed tracks)
and their replay differences remain JSON `null`; plots label them `n/a`.

Outputs include same-track and different-track Euclidean embedding-distance
distributions, true-pair candidate recall versus radius at fixed saved k, and
recall versus k at the saved radius. Both scans also report the fraction of
true tracks fully connected. Track connectivity uses only true edges within
that track, excluding paths through other tracks or noise. Pair recall is pooled
over unordered true pairs; connectivity is pooled over truth tracks. Distance
distributions are pair-weighted; track spans and MST connection radii give one
entry per track. The k-neighbor proposals are symmetrized by union as in training.

The candidate, filtered and GNN-selected graphs each receive a perfect-edge-label
oracle: remove all false/background edges, then take connected components with
the saved minimum component size. This measures that particular graph-constrained
reconstruction; it is not a bound on every possible reconstruction from the
original features. Signal ARI and global track purity/efficiency use the frozen
shared evaluator. Per-event and per-track diagnostics remain with external results.
CPU replay may differ slightly from saved GPU inference; a full validation run
records those differences and the original candidate counts explicitly.
## FM4NPP architecture and 300-epoch control

For an old-data architecture and training-duration control, use
`configs/eggnet_fm4npp_100k_300epochs.json` with `--method eggnet` and the
unchanged 100,000-training-event cohort. Its 30,000,000 updates process one event
each, giving exactly 300 passes through that cohort. The recipe restores the
FM4NPP Appendix C architecture: four graph iterations, four message-passing
rounds per iteration, ten graph neighbors, 64-wide MLPs with two hidden layers
(including the decoder), and a 24-dimensional output embedding. See the
[paper](https://arxiv.org/html/2508.14087v1) for the published architecture.

This control preserves the corrected EggNet objective, seed, optimizer,
validation subset and interval, checkpoint-selection rule and clustering grids
from the historical label sweep. The cosine schedule spans the longer update
budget. It retains CLAS12 coordinates and normalization; it does not reproduce
sPHENIX data preprocessing or substitute the paper's fixed DBSCAN parameters.
The filename's 100k refers to labeled events, not tracks. A different training
cohort size changes the number of passes, so verify its population before launch.
Use fresh external outputs; retain the last checkpoint for explicit resume and
keep results and comparisons outside Git.

## Sweep a frozen Exa.TrkX GNN cut

```bash
python -m clas12_benchmarks.exatrkx_threshold_sweep --run "$RUN_ROOT/exatrkx" \
  --cohort "$RUN_ROOT/cohort/cohort.json" --device cpu --threads 4 \
  --output "$RUN_ROOT/gnn_threshold_sweep"
python scripts/plot_exatrkx_threshold_sweep.py --input "$RUN_ROOT/gnn_threshold_sweep"
```

This validation-only tool holds the saved embedding, candidates, filter, GNN
weights, evaluator and data fixed. It verifies their fingerprints and computes
the GNN scores once per event. Only the final probability cut varies; an edge
survives when `score >= cut`. The saved cut is always included. Override the
default scan with `--cuts`, or use `--max-events` for a smoke check. No parent
checkpoint or selection file is changed. Use a fresh external output directory.

The sweep uses the same cohort fingerprints and inference/reader source checks
as the diagnostic above, including native HDF5 and externally referenced legacy
arrays. Undefined metrics remain `null` and appear as gaps in the threshold plots.

Reported F1 is `2 * pooled matched tracks / (pooled truth + predicted tracks)`;
purity and efficiency use those same pooled counts. Signal ARI and merge rate
are event means from the frozen evaluator. Full connectivity requires all hits
of a true track to connect through same-track edges alone. Clean connectivity
additionally requires that the actual reconstructed component contain no hits
from another signal track. Background contamination is allowed by this clean
definition and is reported separately. Recovery counts show both newly
connected tracks that remain unmerged and previously clean tracks now touching
a merge, relative to the saved cut.

`threshold_curve.json`, `summary.json`, `per_event.jsonl` and `provenance.json`
contain the scan, descriptive validation selections, paired event bootstrap
intervals and provenance. Intervals are conditional on the frozen weights and
do not correct for selecting a cut on this same validation population. No test
events are evaluated. `edge_scores.npz` caches local-index filtered edges and
scores, event offsets and point counts in validation order; identities are in
`per_event.jsonl`. Keep all outputs and plots with external run artifacts,
outside Git. Plotting requires Matplotlib.

## Check upstream mathematical compatibility

```bash
python scripts/verify_sources.py
python scripts/check_tracking_upstream.py --device cpu \
  --output "$RUN_ROOT/upstream_check.json"
python scripts/check_exatrkx_tf_port.py --output "$RUN_ROOT/tf_port_check.json"
```

The first numerical tool checks EggNet with independent neighbor search and
message reductions, Exa.TrkX MLPs, source-derived loss kernels with the declared
class-mean reductions, and connected-component extraction on equivalent retained
edges. Use `--device cuda` to check the Torch math on a GPU. The tool requests
full FP32 matrix multiplication for its independent-reference comparison.
The optional TensorFlow check executes the pinned training loss as well as the
GNN forward, input gradients and parameter gradients at multiple processing depths.

These checks do not equate the CLAS12 training procedure with a published recipe:
full-pair/class-mean objectives, bidirectional GNN graphs, omitted cell features,
coordinate-only filter inputs (`emb_channels=0`), continuous radius,
configurable cuts and singleton rejection are explicit
adaptations. The pinned embedding code defines LayerNorm but leaves it unused,
whereas the Exa.TrkX paper describes LayerNorm. The pinned filter width is also
a source configuration rather than an assertion of exact publication settings.
Keep numerical verification output externally with the run provenance.

## FM track-finding evaluation

New tracking runs resolve the following setting and save it with provenance:

```json
{
  "finding_evaluation": {
    "profile": "mc_reconstruction_v1",
    "truth_policy": "primary_mc_min4_v1",
    "matching": {
      "background_label": -1,
      "iou_threshold": 0.5,
      "min_purity": 0.5,
      "min_efficiency": 0.5
    }
  }
}
```

`--metrics` is now optional. The default is the bundled, unmodified FM reference
at committed revision `295a4e941dc8102b78df2af2f1099c009681d124`, with source paths
and hashes in `fm_reference/sources.json`. An explicit `--metrics` snapshots
another scorer; the MC profile requires its versioned reconstruction API. No FM
model/trainer or uncommitted optimizer/Dice study is imported.

The finding-only view counts owned clusters in each complete event. Segments
with fewer than four become non-target (`-1`); exactly four remain eligible.
It preserves every cluster, original point identity, COATJAVA prediction and
physical provenance. This view supplies **training, validation and test** finding
targets for both HDF5 and ragged data. Raw cohort files, generic readers,
momentum/PID targets, noise targets and noise execution paths retain their
existing behavior. Corrected finding targets change supervision; a newly trained
run is not a continuation under the old label policy.

The MC profile augments the historical accepted matcher with independent
one-to-one Dice association. `mc_track_dice_mean` divides the sum of associated
Dice scores by **all eligible truth tracks**, assigning zero to misses. It gives
equal weight to tracks, rather than events or clusters. Separate non-target
predictions do not reduce primary Dice; non-target clusters inside associated
predictions do. `mc_hit_completeness_global` and `mc_hit_purity_global` pool cluster
counts using those Dice associations. The derived `mc_hit_f1_global` pools the
same hit counts and is generally different from mean track Dice.

`track_efficiency_global` continues to use the independent accepted matcher.
Each pair must pass the recorded purity, completeness and IoU cuts (all 0.5 by
default). Raising cuts reruns accepted matching and does not alter Dice
association. Accepted-track F1, MC hit F1 and mean track Dice have distinct
definitions. Historical unsuffixed diagnostics remain event means; new global
ratios pool counts. Undefined ratios are JSON `null`.

HDF5 provenance joins consume `background_only`, `ownership`, `origin_noise` and
`purity` where available. Only verified pure-overlay clusters enter overlay
contamination/leakage numerators. Excluded MC is not overlay. Missing provenance
remains unknown, with coverage and lower bounds supplied by the FM scorer.
Additional sidecar fingerprints are recorded separately so existing cohort
content hashes retain their historical interpretation.

Summary metadata records `finding_metrics_version: 2`, the effective
`track_eligibility_contract`, the MC profile, accepted-match cuts and a hash of
evaluated physical events/cluster membership, truth, COATJAVA and provenance.
This evaluation policy is separate from a checkpoint's historical training
policy. Audits, frozen evaluations, graph diagnostics and threshold sweeps use
the saved contract; comparisons reject different protocols, scorer/eligibility
source fingerprints, or populations.
New runs also export `per_event_coatjava_metrics.jsonl` on the same support.
Label-sweep plots display MC Dice, hit purity/completeness and accepted-track
efficiency. Bootstrap intervals resample events and recompute pooled counts.

Checkpoint and inference-cut **selection rules are not changed to Dice**.
Tracking-score recipes still select EggNet/GNN checkpoints and final clustering/
GNN cuts by validation accepted-track F1; embedding/filter stages retain their
declared loss/edge criteria. Provenance records these choices. The standard FM
trainer's ARI selection is a separate training-protocol difference to report.
Test labels do not select weights or cuts. To reproduce historical supervision
in a fresh run, set `finding_evaluation` explicitly to
`{"profile": "legacy", "truth_policy": "as_saved"}`. Existing saved provenance
without this setting retains its historical interpretation.

### Rescore existing assignments without retraining

```bash
python -m clas12_benchmarks.rescore_finding \
  --run /absolute/path/archived/eggnet \
  --cohort /absolute/path/archived/cohort/cohort.json \
  --output /absolute/path/new_evaluations/eggnet_mc_min4
```

This CPU-only command validates original cohort/data, prediction event/point
identities, evaluator and checkpoint hashes. It does not import Torch, deserialize
weights, infer predictions or recalibrate thresholds. The output must be a new
directory outside the archived run. It writes new summaries, per-event model and
COATJAVA scores, scorer/eligibility snapshots, and a receipt with parent/prediction hashes, scorer identity,
truth policy, relabelled-cluster count and provenance fingerprints. Original
results are never overwritten; failed attempts do not publish a summary and
require a fresh output path.
For materialized legacy cohorts, output must also be outside the cohort directory:
its historical fingerprint includes every nested file. A model-only legacy run
can be rescored without COATJAVA; its reference event list is empty and no
model-minus-reference comparison is emitted.
Scorer snapshots are compiled directly from their hashed source bytes. Existing
bytecode caches are ignored, and loading an archived scorer does not create new
cache files inside that archive.
For a scoring-only installation use `python -m pip install -e '.[evaluation,hdf5]'`;
Torch and graph packages are unnecessary for this command.

Use `--truth-policy as_saved` for explicitly labelled historical-denominator
rescoring. Use `--min-purity`, `--min-efficiency` and `--iou-threshold` to specify
another reconstruction-quality requirement. Rescoring an old checkpoint under
new truth does not claim it was trained using the corrected supervision.

`evaluate_frozen` retains the parent's original scorer and contract. The tuned
comparison runner audits that historical evaluation, then writes a separate
`original_rescored/` evaluation using the tuned run's common scorer/contract
before comparing. Keep old runs on their matching source snapshot for resume or
strict frozen diagnostics; source checks are not bypassed for migration.

### Match the FM event population exactly

Counts, seeds or dataset paths alone do not establish equivalent populations.
Import the FM workflow's complete `cohorts.json` (`train`, `validation`, `test`)
instead of drawing another random cohort:

```bash
python -m clas12_benchmarks.prepare_finding_cohort \
  --data-root /absolute/path/cvt_cluster_hdf5 \
  --fm-cohorts /absolute/path/fm_run/cohorts.json \
  --plan-id EXPECTED_PLAN_ID_FROM_FM_DATA_CONTRACT \
  --output /absolute/path/new_cohorts/fm_finding
```

Rows must identify whole events with `source_event_index`, `source_id` and
`original_ordinal`. The importer checks the explicit publication plan and physical
identities, preserves order and membership exactly, rejects duplicates/segment
samples, and fits normalization on the imported training coordinates only. It
records eligible-track/cluster counts without changing raw shards. Check that
the plan and source publication match the FM report being compared.

### Parity and regression checks

`test_finding_evaluation.py` covers independently expected split/merge/miss,
short-track, contamination, quality-cut, aggregation and unknown-provenance
cases. When a sibling FM checkout is available (or `CLAS12_FM_ROOT` is set), it
also loads **committed Git objects**, verifies the reference hashes and compares
event/global outputs and the FM offline rescorer directly. It ignores that
checkout's working tree. Rescoring tests verify archive preservation and reject
identity/checkpoint corruption. Both tracking models receive bounded CPU
train/reload/audit/diagnostic integration checks. Bootstrap tests verify pooled
counts rather than averages of per-event Dice or hit ratios.
