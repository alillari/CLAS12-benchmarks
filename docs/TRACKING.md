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

## Train and evaluate

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
