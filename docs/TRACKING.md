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
  --config configs/mild_tuning_20261103.json \
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

`configs/mild_tuning_20261103.json` enables the upstream EggNet Euclidean
squared-hinge objective, warmup/cosine learning rate, and tracking-score checkpoint
selection for EggNet/GNN. The compatibility recipe `configs/first_real_runs.json`
retains the earlier class-balanced pair objective. These are explicit choices;
check the objective and selection rule when comparing runs.

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
  --config configs/mild_tuning_20261103.json \
  --metrics /absolute/path/CLAS12/train/downstream/track_finding_metrics.py \
  --output "$RUN_ROOT/comparison"
```

This trains the configured pipeline, evaluates the parent pipeline with frozen
weights and cuts, audits both, and writes `comparison.json` with paired event
bootstrap intervals. Parent training identities, normalized inputs, truth,
architecture sources and checkpoint hashes must remain compatible. Frozen
inference can also be invoked through `python -m clas12_benchmarks.evaluate_frozen`.
All comparison outputs stay under the external output directory.
