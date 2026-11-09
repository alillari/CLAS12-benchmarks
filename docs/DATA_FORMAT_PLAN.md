# Cluster HDF5 support implementation plan

Add native HDF5 event access to `codex/new-data-format` while preserving existing
RaggedMmap cohorts, frozen momentum manifests, predictions, and score replay.
The selected new input is `cvt-clusters-h5-v1`. Support its native event,
particle, and reconstruction fields through a shared storage interface.

Cluster HDF5 support is implemented on this branch. The design and review
boundaries below describe its contract; the testing plan records qualification.
The [testing plan](TESTING_PLAN.md) defines the current verified baseline,
independent expected results, and acceptance gates for each implementation stage.

## Data products inspected

The products live under this directory on EdgeExperts:

```text
/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/
```

| Product directory | Format | Observed availability |
|---|---|---|
| `mmap_canonical_loose_truthseg_event_v6_02` | RaggedMmap with metadata and JSONL identities | a02c and a518 |
| `cvt_cluster_hdf5_20261103_002` | `cvt-clusters-h5-v1` | a02c and a518 |

The cluster HDF5 manifest is complete and lists 500 parts over 1,999 sources:

| Source split | Parts | Events |
|---|---:|---:|
| `train` | 400 | 13,327,694 |
| `validation` | 50 | 1,665,982 |
| `test` | 50 | 1,667,083 |

The cluster manifest and schema hashes match between hosts. All manifest-listed
HDF5 paths exist on the inspected hosts. Boundary probes read the first and last
event of the first and last part in each split, checked part attributes, finite
coordinates, and owner-to-particle joins. Sampled cluster identities and point
counts agree between hosts in all three splits.
These are inventory and sampled read checks, not full data-content verification.

For the first 512 training events, cluster counts range from 81 to 388, with
median 142. This bounded sample establishes the need to check memory and graph
pair counts before scaling.

The producer's `README.md`, `schema.json`, `recipe.json`, and standalone
`cvt_skim_reader.py` define the field contract. Inspection receipts remain outside
the checkout in the Edge MCP jobs
`benchmarks-data-boundary-inspection-{a02c,a518}-20261109-v1/output/inspection.json`.

## Differences the adapter must preserve

| Concern | Existing RaggedMmap path | HDF5 path |
|---|---|---|
| Storage | Separate `features_<split>` and target directories | Manifest-ordered parts with typed named datasets |
| Event boundaries | One ragged array entry per event | Each domain has its own int64 offsets |
| Source splits | `pretrain`, `test`; cohort preparation takes validation from `pretrain` | Published `train`, `validation`, `test` |
| Event identity | JSONL `source_file,event` | Inventory plus `events.source_id,events.original_ordinal` |
| Measured coordinates | Cluster centers in cm | `clusters.xyz` in cm from first-pass cluster banks |
| Track membership | `seg_target`; -1 is background | `primary_segment`; -1 remains background |
| Particle association | Targets repeated on points | Join positive `primary_tid` to event-local `particles.tid` |
| PID | Small class IDs described by metadata | Signed PDG values with separate validity masks |
| Noise | Existing integer sidecar | Named label and validity fields; cluster contested values can be -1 |
| Momentum target | Finite per-component mean of entrance values, converted from MeV to GeV | `particles.observed_momentum` already in GeV/c, with elementwise validity |
| Reconstruction reference | Per-point `coatjava_seg_pred` and auxiliary columns | `reco_segment`, native track rows, trajectories, and contributor evidence |

The new cluster product uses first-pass BST/BMT clusters and stricter ownership
than the old v6_02 product. It is a new physics population, not a byte-equivalent
repack. Preserve each product's published labels and fingerprint its target
policy; do not claim historical score parity across these populations.

Cluster `energy` is detector-specific signal amplitude, not a shared MeV
calibration. Measured inputs must never include truth, ownership, or COATJAVA
reconstruction fields by accident.

## Compatibility decisions

1. Keep `clas12_baseline_cohort_v1` and the default reader as
   RaggedMmap. Keep the legacy preparation split selection, normalization,
   identities, output schemas, and momentum reductions unchanged.
2. Introduce `clas12_baseline_cohort_v2` for HDF5-backed cohorts. Require an
   explicit storage format, cluster token kind, source split, input feature recipe,
   normalization, label policy, and source fingerprints. Reject format/token
   mismatches and ambiguous roots; an HDF5 error must not trigger legacy fallback.
3. Add an optional `hdf5` dependency extra and lazy `h5py` imports. Legacy loading
   and scoring must continue to work without that extra.
4. Keep the current `Event` features and point order usable by the graph models.
   Add explicit masks and measurement context where needed. Preserve every token
   and `original_point_index`; any model ordering must have an invertible mapping.
5. Use `(source_manifest_id, source_id, original_ordinal)` as the HDF5 event key.
   Record source file hashes from the inventory for overlap checks across copied
   or different inventories. Keep native RUN event IDs separately: in inspected
   data, ordinal 0 has native event 1. Never substitute one for the other.
6. Retain `primary_segment`, positive particle tid, native track ID, and bank row
   as distinct identifiers. For new momentum records use the physical event key
   plus particle tid, retaining the compact segment label as membership metadata.
7. Version HDF5 prediction identities and momentum packs where their semantics
   differ from v1. Keep all existing v1 readers and CSV replay available. Require
   the intended cohort/pack fingerprint when importing new predictions.
8. Keep cohorts within one measurement and target contract by default. Any
   deliberate combination of products needs an explicit provenance mapping and
   compatible policies; a shared array shape is insufficient.

## Implementation sequence

### 1 Separate storage from event projection

Add a small event-source interface under `src/clas12_benchmarks/` with event
count, identity lookup, named field access, contract reporting, selected-content
fingerprinting, and explicit close/context-manager methods. Move current array
opening and JSONL access into a legacy implementation without changing its
outputs. Route `Cohort` through this interface and expose dataset specifications
through a public helper rather than making callers unpack its internal tuples.

Implement a standalone HDF5 source following the producer contract. Build global
indices from manifest order and event counts, then binary-search part boundaries.
Read slices using the offsets of the requested domain, including nested reco
domains. Validate relative paths, part uniqueness, completion, format, plan ID,
part key, event count, offsets, field types, and row alignment. Open files only in
read mode with a bounded LRU; close inherited handles after a process change and
exclude open handles from pickled worker state.

Keep the benchmarks package independent of the foundation-model repository.
`fm4npp/datasets/cvt_hdf5.py` and `cvt_downstream.py` are useful compatibility
references, not new runtime imports.

**Acceptance:** current legacy event outputs remain equal; HDF5 indexing works at
every split and part boundary without creating or modifying source files.

### 2 Define label and feature projections

For cluster tracking, map `clusters.xyz` to measured xyz,
`clusters.primary_segment` to `seg_target`, and `clusters.reco_segment` to the
stored COATJAVA grouping. Preserve -1 as background for track finding.

Keep raw noise labels and masks accessible. When matching the foundation-model
adapter, explicitly select its `primary_unassigned_is_noise_v1` policy:
`noise_target = (primary_segment < 0)`, including contested unassigned clusters.
A recipe using only valid native noise labels must name that different policy
and apply `primary_noise_valid`; casting raw -1 to a boolean is invalid.

Retain signed PDG labels and `primary_pid_valid`. Provide an explicit mapping to
the existing pion/kaon/proton/electron/other classes when requested, with an
ignore value for invalid rows. Record the mapping in the cohort contract.

Start cluster benchmarks with the existing position-only input transformation.
Additional measured energy, time or geometry features require named recipes,
training-only normalization, and a recorded feature order. Keep validity masks
aligned through selection, sorting, batching, and prediction export.

**Acceptance:** named fields, converted targets, masks, and point order agree
with independent synthetic expectations and direct reads of sampled events.

### 3 Prepare cohorts and label sweeps from either source

Update `prepare_cohort.py` and `prepare_label_sweep.py` to use the source interface.
For HDF5, sample training only from `train`, validation only from `validation`,
and evaluation only from `test`. Preserve the legacy split policy for v1 inputs.
Freeze explicit members, seeds, ordering, nested budgets, fixed holdouts, and
training-only normalization. Use chunked identity reads rather than a Python
loop issuing one HDF5 read per event across millions of rows.

Write small HDF5-backed cohort manifests referencing the original shards;
avoid mandatory conversion or copying the whole product. Preserve the existing
legacy materialized-subset behavior. Extension must verify the parent source,
retain its training and validation identities, and exclude old evaluation events
when drawing a fresh test cohort.

Update provenance in `tracking.py` and `evaluate_frozen.py`: their current walk
of the cohort directory cannot fingerprint externally referenced HDF5 data.
Record source manifest/schema/recipe/geometry identities, selected event and
consumed-content hashes, feature/label policies, and normalization. Distinguish
these from full shard checksums. Resume must reject source or contract drift.
Preserve the existing legacy provenance verification path for archived runs.

**Acceptance:** nested budgets and holdouts are stable, duplicate physical events
are rejected across aliases/products, and source changes invalidate reuse.

### 4 Add HDF5 momentum preparation and versioned scoring

Keep `prepare-momentum --prepared ...` and `clas12_momentum_tracks_v1` as the
frozen legacy path. Add an HDF5 preparation route that freezes its own track keys
and target contract, rather than applying old raw-row keys or hashes to new data.

Join each primary segment through its unique positive tid to the event's particle
row. Use valid finite `observed_momentum` directly in GeV/c. Require all three
components for the vector benchmark and report exclusions; never multiply these
values by 0.001, average in generated momentum, or substitute missing components.
Record the observed reference definition because it differs from the old target
construction.

Make multiplicity cuts explicit in the new pack. Keep the old 12–100 cluster
contract for v1. Make new HDF5 cluster cuts a declared recipe rather than
silently demoting shorter native segments. Separate event cohorts from
task-specific track eligibility and count excluded tracks.

For CVT comparisons, declare the reconstruction pass and association policy.
Resolve event-local contributor evidence and candidate rows, retaining missing
and ambiguous cases. A conservative initial reference requires one qualified,
unique-owner candidate with valid benchmark p/theta and same-pass entrance phi;
otherwise mark CVT missing and retain the model track. Pin the qualification and
purity thresholds in the recipe, and record coverage counts.
Any alternative candidate-selection rule must have a separate contract; do not
reuse the legacy first-finite-auxiliary-row rule on native reco tables.

Extend `TrackManifest`, `momentum_evaluation.py`, CLI inspection, and prediction
alignment to dispatch by schema. Reuse the existing metric calculations for
completely valid vectors, while preserving the legacy companion audit path.

**Acceptance:** HDF5 truth/identity/point membership is independently verified,
missing CVT references stay explicit, and archived v1 score replay is unchanged.

### 5 Bound memory use for larger event cohorts

Audit `tracking.py`, which currently materializes training and validation events
in Python lists, and all pair/graph construction limits. Use indexed lazy access
with bounded caching before large HDF5 cohorts. Retain explicit failures for
configured graph limits; never silently drop points, sample truth-dependent
tokens, or raise every cap without measuring memory. Read only required field
slices and keep event caches bounded when training, validating, or exporting.

**Acceptance:** cluster prediction exports restore every saved point, and bounded
smoke runs fit the declared CPU/GPU memory and pair budgets.

### 6 Verify compatibility and document both workflows

Apply the gates and numerical criteria in [the testing plan](TESTING_PLAN.md).

Add synthetic HDF5 fixtures with separate offset domains, multiple shards,
noncontiguous tids, contested clusters, invalid PID/momentum, structured native
reco rows, and missing/ambiguous reco associations. Compare an equivalent
synthetic event represented in both storage formats; use real old/new
products only for their own contracts, since their physics populations differ.

Exercise metadata/offset corruption, unknown formats, unavailable fields, source
path escapes, duplicate event keys, reordered predictions, unit conversion,
file-handle eviction, worker pickling, and opening legacy inputs without h5py.
Verify that changing truth or reco fields cannot change measured model inputs.

Run the existing unit suite, source verification, synthetic tracking pipeline,
and frozen legacy momentum replay. Then run bounded real-data reader and
train/save/reload/evaluate/export smoke checks on HDF5 cluster cohorts on both
hosts. Recompute metrics from exported predictions and verify complete identities
and denominators. Keep generated
cohorts, snapshots, reports, and predictions in external artifact storage.

Update `README.md`, `docs/TRACKING.md`, `docs/MOMENTUM.md`, and reusable example
configs with separate working commands for each supported format, dependency
extras, identity schemas, target policies, split rules, and resource limits.

## Suggested review boundaries

1. Storage interface, unchanged legacy behavior, HDF5 cluster reader, and fixtures.
2. Cohort preparation, sweeps, provenance, prediction identity, and tracking smoke.
3. HDF5 momentum packing, CVT association, schema-aware scoring, and legacy replay.
4. Streaming/resource work and documentation. Move required streaming work
   earlier if the first cluster smoke checks need it.

Each boundary should remain usable with old data. Supporting a new file format
does not make its different labels, target definition, or checkpoints
scientifically interchangeable with the old product.

## Implementation status

The storage, projection, cohort/sweep, momentum, schema-aware prediction,
provenance, and bounded event/graph access paths are implemented. Legacy reader
outputs and both tracking recipes reproduce their pre-refactor outputs. The
legacy momentum pack rebuild is byte-identical, and all twelve archived score
replays pass. Real HDF5 workflows have passed on a02c CUDA and a518 CPU with the
same frozen event keys. See [testing and qualification](TESTING_PLAN.md) for
test boundaries, scripts, and evidence locations.
