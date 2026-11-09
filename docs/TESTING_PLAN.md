# Legacy and cluster HDF5 testing plan

Protect every implemented benchmark workflow while adding the cluster HDF5
support described in [the implementation plan](DATA_FORMAT_PLAN.md). Acceptance
requires correct identities, inputs, targets, training behavior, predictions,
metrics, and provenance. A successful process exit or decreasing loss alone is
insufficient.

HDF5 runtime support is implemented on this branch. The baseline below records
the legacy implementation before changes; current implementation qualification
is recorded at the end. The test cases and gates remain the acceptance contract
for subsequent changes.

## Current verification baseline

Checks on November 9, 2026 used the source at `5ce04ba` on
`codex/new-data-format`, with documentation changes only.

| Check | Observed result |
|---|---|
| Pinned upstream source verification | Passed: 46 EggNet, 59 Exa.TrkX, and 51 OneFormer3D files |
| Complete existing unit suite | Passed: 40 tests, no skips |
| Audit tests under `python -O` | Passed: the same three audit tests rerun with assertions disabled |
| Synthetic tracking pipeline, compatibility recipe | Passed for EggNet and Exa.TrkX |
| Synthetic tracking pipeline, corrected-loss recipe | Passed for EggNet and Exa.TrkX |
| Audit of each pipeline's saved predictions | Passed for all four method/recipe combinations |
| Frozen re-evaluation of those four trained pipelines | Passed; original model and COATJAVA summaries reproduced within absolute tolerance 1e-12 |
| Independent audit of frozen outputs | Passed for all four combinations |

The recipes were `configs/first_real_runs.json` and
`configs/mild_tuning_20261103.json`. Each synthetic pipeline used four updates per
stage and three evaluation events. Audits recomputed per-event and summary
metrics, checked every saved point identity, and verified selected checkpoint
hashes. Frozen checks reused the identical cohort to test numerical replay;
expansion to fresh holdouts remains a separate acceptance case below.

Runtime checks ran on a02c in an isolated CPU container with two CPUs and an
8 GiB memory limit. The existing image supplied Python 3.12.3, NumPy 1.26.4,
SciPy 1.15.2, Torch `2.8.0a0+5228986c39.nv25.05`, scikit-learn 1.6.1,
Matplotlib 3.10.3, and h5py 3.16.0. PyG 2.4.0 was installed into the isolated
job's dependency directory. No host environment or primary checkout was changed.

Receipts and logs are outside the checkout on a02c:

```text
/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/
  benchmarks-testplan-baseline-a02c-20261109-v1/output/
    verification.json
    environment.log
    pip-freeze.txt
    unit-suite.log
    optimized-audits.log
    pipeline-*.log
    audit-*.json
  benchmarks-testplan-frozen-a02c-20261109-v1/output/
    frozen-verification.json
    *-audit.json
```

This baseline does not establish clean-install behavior, CUDA execution,
TensorFlow/Sonnet parity in this environment, fresh real-data momentum replay,
or HDF5 correctness. That environment already supplied packages beyond
the declared extras; at that baseline, Matplotlib had no declared plotting
extra. The implementation now declares plotting and HDF5 extras and has passed
the clean-install checks recorded below.

## Coverage and remaining obligations

| Implemented component | Existing evidence | Additional acceptance coverage |
|---|---|---|
| Legacy event reader and feature construction | `test_port.py`: sidecars, geometry, missing files, duplicate events, metadata drift, point order | Before/after compatibility oracle, all supported layouts, invalid geometry, optional-dependency isolation |
| Cohort preparation and extension | `test_tracking.py`: selected roles, copied arrays, preserved training, expanded validation, fresh test | Deterministic re-preparation, failure cleanup, drift during extension, CLI failure behavior |
| Nested label sweeps | `test_label_sweep.py`: nesting, fixed holdouts, per-budget training normalization | Invalid budgets, source drift, reproducible member order and summary counts |
| EggNet and Exa.TrkX | `test_tracking.py`: loss values/gradients, neighbors, updates, strict reload; complete synthetic pipelines | GPU smoke, full pipeline interruption/resume, model-input isolation, bounded large events |
| Torch Interaction Network | NumPy block calculation and gradient/execution checks | Existing TensorFlow/Sonnet parity script when the port changes |
| Four semantic graph operators | `test_port.py`: GATConv, GCNConv, GraphConv, SAGEConv update/reload | Explicit feature/class/ignore-label contract when a task-specific trainer is added |
| Predictions and audits | `test_port.py`, `test_tracking_audit.py`, eight successful pipeline/frozen audits | Full tamper matrix and schema dispatch for HDF5 identities |
| Frozen re-evaluation | Four current same-cohort integrations | Fresh-test extension; changed training, evaluator, source, or checkpoint must be rejected |
| Comparison, bootstrap, and plots | `test_comparison.py`, `test_label_sweep.py`, `test_sweep_plot.py` | Determinism, deliberately unequal denominators, paired-population checks, representative visual review |
| Momentum packing and scoring | Six `test_momentum.py` cases: membership, units, nesting, drift, keyed scoring, companion audits | CLI round trips, additional analytic metric cases, archived real-data replay, HDF5-specific contract |
| Imported source | All 156 pinned files verified | Repeat after vendor or source-lock changes |

OneFormer3D currently has an upstream source import, not a completed CLAS12
adaptation. External momentum trainers described in `MOMENTUM_PLAN.md` are also
future work. Their source/planning status must not be reported as a passing
training workflow.

## Independent expected results

Use a small, hand-authored logical event fixture as the source of expected
identities, coordinates, masks, groups, and target vectors. Serialize it
independently as RaggedMmap and HDF5. Do not derive expected HDF5 values by calling
the production adapter, or generate every fixture through a writer that shares
the same mapping code as the reader.

Keep three complementary checks:

1. Compare stored fields with direct NumPy/h5py reads and the field dictionary.
2. Compare projected events with manually specified expected results.
3. Compare equivalent logical events through both backends after deliberately
   aligning their feature and target policies.

Real v6_02 and HDF5 products contain different selections and label definitions.
Test each against its own contract. They are not an oracle for equal real-data
predictions or scores across products.

Before the storage refactor, capture the current legacy reader's selected
events, normalized inputs, labels, point indices, and semantic momentum-pack
contents. Compare the refactored reader with these frozen outputs in the same
environment. Compare packed arrays and metadata rather than ZIP bytes whose
container metadata may vary. Preserve existing cohort-file fingerprints and
archived output schemas.

## Required synthetic cases

### Storage indexing and worker lifetime

Use at least three shards with unequal event counts. Give clusters, particles,
tracks, and trajectories different offsets so an incorrect shared-offset read
cannot accidentally pass. Include zero-length domains, an all-background event,
unequal track sizes, and fields with recognizable per-event sentinel values.

Check first/last events, both sides of every shard boundary, arbitrary read order,
repeated reads, and documented negative-index behavior. Out-of-range and
noninteger indices must fail explicitly. Verify event counts, row shapes, dtypes,
and slices against direct field reads. Empty collections must remain empty;
model-level empty-event eligibility must be explicit and counted.

Open more shards than `max_open`; verify actual handle counts stay bounded and
evicted files close. Exercise context exit, exceptions, pickling, process changes,
and available spawn/fork worker modes. Worker reads must equal single-process
reads, with no shared inherited live HDF5 handles. Record file descriptors before
and after repeated source creation and teardown.

### Corrupt inputs and publication failures

Alter one property per case: unsupported format/schema, incomplete manifest or
part, plan ID, split, part key, event count, duplicate path, missing field, wrong
dtype, wrong vector width, absent file, truncated data, or malformed offsets.
Offsets must start and end correctly, be monotonic, have event-count-plus-one
length, and stay within their own domain. Include absolute/traversing paths and
symlinks that escape the allowed dataset root.

Each failure must identify the relevant source/part/field and leave the source
unchanged. A failed preparation must not publish a usable completion receipt,
replace an existing artifact, or leave a partial result that later commands
accept as complete. Check these invariants after an injected mid-write failure.

### Cluster labels and event-local joins

Choose compact segment labels, particle tids, native track IDs, and bank rows
with deliberately different values. For example, segment 3 can belong to tid 17
stored in particle row 1. Reorder particle rows: truth associations must remain
the same. Duplicate or nonpositive particle tids, conflicting segment owners,
or missing positive owners must fail rather than resolve through array position.

Include valid signal, measured background, secondaries, contested ownership,
valid signed PDG values, and invalid PID fillers. Verify that:

- Track finding retains -1 as background and never makes background/background
  pairs positive. Labels `[-1, -1, 7, 7, 21, -1]` have only the pair (2, 3) as
  a positive unordered pair.
- Native noise and validity masks remain accessible. The named
  `primary_unassigned_is_noise_v1` projection includes contested unassigned
  clusters as noise; a valid-native-label recipe excludes them from its loss and
  metric denominator. Invalid -1 must not silently become boolean true.
- PID mapping handles both signs, unknown valid species, and invalid rows
  independently. Invalid PID is not the valid "other" class.
- Coordinates, labels, masks, optional measurements, and original indices remain
  aligned through every ordering and export operation.

### Units, momentum, and reconstruction references

Use a known observed vector `(0.3, -0.4, 1.2)` GeV/c. Its pT is 0.5 GeV/c and
its magnitude is 1.3 GeV/c. The corresponding legacy vector is
`(300, -400, 1200)` MeV/c. Verify the conversions independently; the HDF5 vector
must never undergo the legacy factor of 0.001.

Choose generated momentum different from observed momentum. Give invalid
observed components finite filler values in some cases and NaNs in others.
Vector eligibility must require all components both valid and finite, report
exclusions, and never fall back to generated values. Test track lengths just
below, at, and above each declared cut; preserve the v1 12–100 cluster bounds.

For legacy data, preserve the finite per-component truth mean and first jointly
finite auxiliary CVT-row behavior. For HDF5, test the chosen association policy
with no candidate, one qualified candidate, ambiguous candidates, insufficient
purity, invalid kinematics, and missing trajectories. Use different CVT/CVTRec
track IDs, rows, and entrance angles to catch cross-pass joins. Give entrance phi
and DOCA phi0 different values. Missing CVT removes a track only from the common
reference subset, not from the all-model-track population.

For scoring, include perfect predictions, a known component error, a scalar pT
offset of 0.05 GeV (50 MeV error), wrapped phi near ±180 degrees, zero transverse
momentum, missing CVT for every track, and an empty common subset. Compute
expected values with direct equations and counts independent of the evaluator.

### Cohorts, leakage, and provenance

Verify repeated preparation with the same seed produces the same selected
identities and order. For HDF5, roles must use the published train, validation,
and test splits; legacy v1 keeps its existing split rules. Check nested budgets,
fixed holdouts, extension behavior, and exact reported event/track/point counts.

Detect duplicate physical events across aliases and relocated roots. If mixing
inventories is supported, test remapped source IDs against original-source hashes
and ordinals. Reject incompatible measurement/target contracts by default.
Unknown identity correspondence must not be treated as verified disjointness.

Change only validation/test coordinates: training normalization must not change.
Change test labels or predictions: selected checkpoints and validation-selected
cuts must not change. Freeze the selection before final test evaluation.

Change each fingerprinted input in turn: source manifest, schema, recipe,
geometry, selected feature content, target content, feature order, label policy,
normalization, metric code, and checkpoint. Reuse or resume must reject relevant
drift. An unchanged JSON manifest cannot excuse changed consumed HDF5 content.
Relocation with identical content must preserve the physical keys; document
whether path changes require a new cohort file/fingerprint.

### Model inputs, training, and recovery

Mutate truth, PID, ownership, particle targets, and COATJAVA fields while keeping
measurements fixed. Measured feature tensors, geometry-neighbor construction,
and deterministic inference must remain unchanged. Supervision may change;
model inputs must not. Check duplicate coordinates and tiny events explicitly.

For EggNet, Exa.TrkX stages, and all four graph operators, require finite loss,
finite gradients, a parameter update, strict state reload, and unchanged
evaluation outputs after reload. Retain independent loss/gradient calculations.
Check that graph creation does not truncate tokens or add self edges contrary
to its declared policy.

Compare uninterrupted training with interruption and resume across an epoch
boundary and validation checkpoint. Verify model, optimizer, learning-rate
schedule, shuffle/RNG state, update count, selected checkpoint, and thresholds.
The existing isolated trainer tests remain mandatory; add the complete runner
case with the same planned total update budget. Production integrity checks must
still execute under `python -O`; use explicit exceptions rather than `assert`.

### Prediction round trips and metric audits

Round-trip CLI preparation, inspection, export, and evaluation for every
supported schema. Shuffle prediction rows: keyed metrics must be unchanged.
Reject duplicates, missing or unknown tracks/events/points, wrong method/schema,
wrong cohort/pack hash, invalid shapes, nonfinite values, and wrong units.

Tamper independently with a prediction, point index, cached metric row, summary,
evaluator, or selected checkpoint. The audit must detect disagreement rather
than trusting cached summaries. Check both CSV and gzip momentum predictions.

Recompute tracking metrics on hand-designed perfect, split, merged, all-noise,
and empty-prediction cases. Verify undefined values and distinguish global count
ratios from means over events. For example, matched/true counts of 1/1 and 1/9
give global efficiency 2/10, not the average of 1 and 1/9. Paired bootstrap must
resample the same event indices for both methods, retain defined metrics when
another is undefined, and reproduce intervals with a fixed seed.

Render representative sweep plots with missing metrics, a subset of budgets,
and long labels. Check units, counts, legend meanings, uncertainty intervals,
and clipping in addition to successful file creation. Different evaluation
populations or COATJAVA baselines must not be presented as a paired comparison.

## Numerical pass criteria

| Quantity | Required comparison |
|---|---|
| Identities, labels, masks, indices, dimensions, counts, hashes | Exact equality |
| Stored field reads | Exact values/dtypes and identical NaN masks |
| Legacy outputs before/after storage refactor in the same environment | Exact identities, order, raw/projected arrays and semantic packed contents |
| Independently calculated float32 feature transforms | Start with rtol 1e-6, atol 1e-7; units fixed and special cases explicit |
| Same-environment deterministic CPU resume | Bitwise parameter equality, as required by current resume tests |
| NumPy float64 Interaction Network reference | Existing rtol 1e-10, atol 1e-12 |
| TensorFlow/Sonnet forward probabilities | Existing rtol 2e-5, atol 2e-6 |
| TensorFlow/Sonnet input and parameter gradients | Existing rtol 2e-4, atol 2e-6 |
| Tracking audit against saved metrics | Existing rtol 0, atol 1e-12; identical undefined-value masks |
| Archived momentum table replay | Counts exact; existing tail comparison tolerance 1e-12; continuous scores rtol 2e-6, atol 2e-5 in the reported unit |
| Archived native pT scores | Existing rtol 2e-6, atol 2e-5 MeV; identical denominators |

Each test must name its unit and comparison tolerance. Cross-device GPU checks
need tolerances established from a frozen reference in the tested environment;
they must not weaken identity, count, mask, or unit checks. A unit-scale or
association error is a failure, not a numerical tolerance adjustment.

## Test environments and acceptance gates

| Gate | Run when | Required evidence |
|---|---|---|
| G0 Clean installation | Before the first code refactor; repeat for dependency changes | Fresh base-only, tracking, graphs, plotting, and HDF5 environments; supported Python minimum 3.10 and Edge Python 3.12; documented install/CLI imports work without undeclared dependencies |
| G1 Legacy regression | Every implementation change | All existing tests, source verification, before/after legacy oracle, both synthetic tracking recipes and prediction audits; no unexplained skips |
| G2 HDF5 storage and projection | Reader/contract milestone | Independent fixtures, corrupt-input matrix, joins/units/masks, schema-aware CLI tests, worker lifetime, legacy operation without h5py |
| G3 Training and momentum integration | Cohort/trainer/pack milestone | Nested cohorts, full runner resume, both tracking methods, four graph operators, HDF5 momentum pack/score, output audits, negative provenance checks |
| G4 Real data and device qualification | Before declaring HDF5 support ready | Reader qualification on both hosts, bounded CPU/CUDA workflows, same frozen event keys, resource measurements, archived legacy replay |
| G5 Release review | Before merging support | All applicable gates passed, no required case recorded merely as skipped, tested commands and receipts linked, remaining limitations explicit |

G0 must prove legacy imports and legacy pack/scoring work without h5py. The full
current suite needs graph and plotting dependencies; split dependency-specific
test jobs instead of pretending that the complete suite is a base-only check.
Add the proposed HDF5 extra and declare the plotting dependency before using
those extras as installation recipes.

Automate G0–G3 in CI or a reproducible container harness. Use the standard-library
unittest runner already used by the repository. Suggested additions are
`test_legacy_compatibility.py`, `test_hdf5_reader.py`, `test_hdf5_cohort.py`,
`test_hdf5_momentum.py`, and `test_cli_contracts.py`; these are proposed files,
not tests that already exist. Parameterize by backend where behavior is shared.
Keep worker processes, threads, random seeds, sample counts, and time limits
bounded. Test counts or line-coverage percentages alone are not acceptance.

Run the existing TensorFlow/Sonnet parity script when Interaction Network code,
bindings, upstream pins, or its verification dependencies change. A missing
optional environment may be recorded for unrelated data-only changes; a port
change cannot pass its gate by skipping its parity check.

## Real-data qualification and resource checks

On both hosts, inspect all 500 cluster shards at metadata level: publication
attributes, split membership, required field shapes/dtypes, every domain's
offsets, terminal lengths, and source inventory. Read at least the first and
last event of every shard plus 16 deterministic, uniformly selected events per
shard. Group reads by shard and compare required fields with direct h5py reads.
Report distinct sampled events and cases absent from the sample; synthetic
fixtures must still cover contested, invalid, empty, and ambiguous cases.

Use manifest-frozen, bounded real cohorts for train/save/reload/evaluate/export
checks. A starting smoke budget is 16 training, eight validation, and eight
evaluation events, four updates per tracking stage. Freeze all keys and use the
same keys on a02c and a518. Use labels only for supervision/eligibility and
stratified diagnostics, never for model input construction. Run CUDA checks on
an available GPU without interrupting existing training.

Measure cold/warm event-read time, throughput, peak process memory, peak CUDA
memory, open handles, graph sizes, and actual processed points. Add a bounded
soak of 10,000 reads across more shards than the handle cache can hold. With
fixed batch/cache limits, resident data must not grow with total source size;
live handles must stay within the configured bound and return to baseline after
closing. Record warm allocator behavior separately from live-object leaks.

Test event sizes at, below, and above graph/pair limits. Either process all
points or return the documented explicit resource error; silent truncation is
always a failure. Freeze hardware-specific memory/time budgets in the receipt
before a scale check rather than choosing thresholds after seeing a result.

Rebuild the legacy momentum pack from its frozen companion keys and rerun all
available archived score-table and native-pT checks before accepting momentum
refactoring. Verify exact populations and the existing tolerances above. This is
replay of saved predictions, not a request for production retraining.

Full dataset checksums and a full scientific evaluation are separate evidence
levels. Sampled reader checks establish sampled correctness; they do not certify
every payload byte or demonstrate model accuracy on the new physics population.

## Commands and evidence to retain

Existing commands in an environment with the required dependencies:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ROOT=/absolute/path/result_deep_storage/benchmarks/validation
export METRICS=/absolute/path/CLAS12/train/downstream/track_finding_metrics.py

python scripts/verify_sources.py
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -p test_tracking_audit.py -v

python scripts/check_tracking_pipeline.py --output "$RUN_ROOT/compatibility" \
  --metrics "$METRICS" --config configs/first_real_runs.json
python scripts/check_tracking_pipeline.py --output "$RUN_ROOT/corrected" \
  --metrics "$METRICS" --config configs/mild_tuning_20261103.json

python scripts/audit_tracking_predictions.py \
  --run "$RUN_ROOT/corrected/eggnet" \
  --cohort "$RUN_ROOT/corrected/cohort/cohort.json" \
  --output "$RUN_ROOT/corrected/eggnet-audit.json"
```

Use fresh output directories. Repeat the final audit for both methods and both
recipes. `PYTHONDONTWRITEBYTECODE` prevents test imports from adding Python
caches to vendor trees that source verification correctly treats as extra files.
The TF parity command is `python scripts/check_exatrkx_tf_port.py --output ...`
in its separately documented optional environment. Legacy momentum replay
commands are in [the momentum guide](MOMENTUM.md).

For each gate retain the repository commit and source snapshot/dirty patch,
image ID and environment versions, exact argv, seeds, hardware/resource limits,
data and cohort identities, evaluator/checkpoint hashes, counts and exclusions,
per-check status, numerical differences, logs, and recomputed audit results.
Distinguish passed, failed, not run, and not applicable. A required unavailable
environment remains an unmet gate. Keep generated receipts and artifacts outside
the checkout; keep reusable fixtures, checks, recipes, and this plan in Git.

## Cluster HDF5 implementation qualification

Initial qualification completed on November 9, 2026. That suite contained 66 tests
(the original 40 plus 26 HDF5/compatibility tests), with no skips, passing on
Python 3.10.22 and 3.12.3 with stable CPU Torch 2.8.0. Audit tests also pass under
optimized Python. Fresh editable installs of all declared extras passed in
both Python environments. A separate clean base-only environment passed 19
legacy/core tests with neither h5py nor Torch installed. Tracking requires the
complete source checkout containing the pinned vendor sources and configs;
the documented installation uses `pip install -e` from that checkout.

| Gate evidence | Result |
|---|---|
| Independent storage/projection fixtures | Passed, including separate offsets, shard boundaries, spawn/fork, masks, joins, zero-length collections, corrupt manifests/parts/offsets, and source aliases |
| Failed publication | Injected a failure after a partial write; no completed output survived and a fresh preparation succeeded |
| Legacy reader oracle | New reader outputs exactly matched the original reader at `5ce04ba` on selected synthetic events |
| Legacy tracking regression | Both recipes and both methods reproduced original predictions and summaries exactly; audits passed |
| Legacy raw momentum rebuild | Byte-identical SHA-256 `4d0c64f688bdc7e96adf1e39a8b4328fc0adbd6812e688aba53b09e0c8b3485f` |
| Archived momentum replay | All 12 runs reproduced saved results, including score-table and native-pT checks; 50,000 tracks / 49,803 CVT-valid per evaluation |
| Real shard qualification | All 500 shards on each host passed schema/shape/dtype/offset checks; 8,999 distinct sampled events per host agreed with direct h5py reads |
| Cross-host data identity | Sampled-content digest and frozen real cohort fingerprint matched exactly between a02c and a518 |
| Reader soak | 10,000 reads per host; at most two open cached parts, stable resident memory after warm-up, file descriptors returned to baseline |
| Synthetic HDF5 integration | CLI prepare/inspect/export/pack/score, both tracking methods, frozen replay and prediction audits passed |
| Full runner recovery | Both methods matched uninterrupted CPU model parameters bitwise after interruption and resume; selections/scores matched and audits passed |
| Fresh frozen holdouts | Both methods passed re-evaluation/audits on new test events; changed training order was rejected |
| Real HDF5 integration | Passed on a02c CUDA and a518 CPU, including export, momentum, tracking, save/reload, frozen replay and audits |
| Source integrity | All 156 pinned upstream files verified unchanged |

Real integration used the same 16 training, eight validation, and eight
evaluation events on both hosts, with four updates per tracking stage. Each
method exported all 1,061 evaluation points. The momentum pack retained 20/4/3
training/validation/evaluation tracks under the declared 12–100 cluster cut and
recorded all exclusions and missing references. Training writes `resources.json`
with wall time, peak RSS, CUDA allocations/reservations when applicable, and
live HDF5 handle counts before context-manager cleanup. Event caches are bounded;
part caches hold at most four files per source split in normal workflows.

The sampled-content digest was
`146072f05bfd9214ffcc4b517ebee0b9bcf7b7c6262bc471bb88d89ebdda123d`.
Real cohorts shared fingerprint
`9ba631155f75c8b0e6eef1213a230a49083564c8a40cf415d78c22fccd078e87`.
These are sampled/selected-content checks, not full shard payload checksums.
The short training runs establish software execution and integrity, not accuracy
or convergence on the new population. a518 GPU training already in progress was
left running; CUDA qualification used the idle a02c GPU.

Successful receipts are under each host's
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/JOB_ID/output/`:

| Job ID | Main receipt |
|---|---|
| `benchmarks-hdf5-clean-install-a02c-20261109-v1` | `install-check.json` |
| `benchmarks-hdf5-editable-install-matrix-a02c-20261109-v3` | `install-matrix.json` |
| `benchmarks-hdf5-final-matrix-a02c-20261109-v2` | `final-tests.json`, `310.log`, `312.log` |
| `benchmarks-hdf5-legacy-regression-a02c-20261109-v1` | `legacy-regression.json` |
| `benchmarks-hdf5-legacy-raw-rebuild-a02c-20261109-v1` | `rebuild.json` |
| `benchmarks-hdf5-qualification-a02c-20261109-v1` | `qualification.json` |
| `benchmarks-hdf5-qualification-a518-20261109-v1` | `qualification.json` |
| `benchmarks-hdf5-recovery-pipeline-a02c-20261109-v1` | `synthetic/verification.json` |
| `benchmarks-hdf5-real-pipeline-a02c-cuda-20261109-v2` | `real/verification.json` |
| `benchmarks-hdf5-real-pipeline-a518-cpu-20261109-v2` | `real/verification.json` |

The archived momentum replay receipt from this implementation is also available
locally at `/tmp/clas12-hdf5-legacy-replay-20261109.json`. Initial install-harness
attempts lacked checkout resources or used a nonportable interpreter alias;
the successful complete-checkout matrix above supersedes them.

Reusable additional checks:

```bash
python scripts/check_hdf5_pipeline.py --output "$RUN_ROOT/hdf5-synthetic" \
  --metrics "$METRICS" --check-recovery
python scripts/check_hdf5_pipeline.py --data /absolute/path/cvt_cluster_hdf5_20261103_002 \
  --output "$RUN_ROOT/hdf5-real" --metrics "$METRICS" --device cuda
python scripts/qualify_hdf5.py /absolute/path/cvt_cluster_hdf5_20261103_002 \
  --output "$RUN_ROOT/hdf5-qualification.json"
```

`.github/workflows/tests.yml` adds the Python 3.10/3.12 CPU install, unit,
optimized-audit, and pinned-source checks. The equivalent local/container matrix
passed; the workflow itself has not been dispatched on GitHub. Full integration
still takes an explicit path to the shared CLAS12 metric module. TensorFlow
port-parity was not rerun because the Interaction Network and its upstream
bindings were unchanged; its independent NumPy and gradient checks remained in
the passing suite.

## Review fixes and final regression checks

The final review integrated `main` through `31d8463`, preserving the merged
geometry-based Exa.TrkX edge orientation, explicit historical `legacy_index`
recipes, corrected defaults, recipe validation, and upstream fidelity checks.
The HDF5 integration check now uses the current `tracking.json` recipe by
default; historical recipes remain explicit regression cases.

Review fixed these gaps and added targeted tests:

- Legacy manifests referencing arrays outside the manifest directory now hash
  their actual selected payloads for resume/audit. Existing fingerprints for
  archived materialized cohorts remain unchanged.
- Missing legacy array directories fail before mmap_ninja can create them in
  the input product. HDF5-only noise policies cannot be silently ignored by the
  legacy preparation path.
- Extending cohorts or preparing sweeps from a relocated HDF5 product verifies
  the parent's selected bytes, rather than accepting matching metadata and
  normalization alone. Identical relocated copies remain supported.
- HDF5 digests handle iterator inputs and recheck live metadata/geometry.
  Batched identity reads handle negative-index aliases consistently.
- Prediction audits recheck consumed-data fingerprints, and CLI/audit readers
  close their resources on failures. Invalid normalized coordinates fail early.
- HDF5 momentum scores cannot be verified against legacy score tables or native
  legacy summaries with a different target contract.

All 80 tests passed with no skips on Python 3.10 and 3.12; all four audit tests
also passed under optimized Python. Both legacy tracking recipes reproduced
their archived predictions and summaries exactly. The current geometry recipe
passed HDF5 full-run interruption/resume, fresh-holdout evaluation, and audits.
Independent upstream checks passed on CPU and CUDA, and the real HDF5 GPU
workflow passed again. All twelve archived momentum score/native-pT replays
still match, and all 156 pinned vendor files remain unchanged.

Review receipts on a02c, using the same job/output prefix above:

| Job ID | Main receipt |
|---|---|
| `benchmarks-hdf5-review-matrix-a02c-20261109-v2` | `final-tests.json`, `310.log`, `312.log` |
| `benchmarks-hdf5-review-integration-a02c-20261109-v1` | `review-integration.json`, `hdf5/verification.json`, `upstream.json` |
| `benchmarks-hdf5-review-gpu-a02c-20261109-v1` | `review-gpu.json`, `real/verification.json`, `upstream-cuda.json` |

The refreshed local momentum replay receipt is
`/tmp/clas12-hdf5-review-legacy-replay-20261109.json`. These review checks use
bounded cohorts and retain the earlier all-shard/read-soak qualification; they
do not claim production model convergence or full payload checksums.
