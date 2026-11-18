# Momentum track manifest and evaluator

Both legacy and HDF5 momentum packs are supported. The original commands and v1
replay contract below remain unchanged; the HDF5 route has its own v2 contract.

The benchmark measures entrance momentum for a single CVT track given its MC-associated cluster membership. It uses the central-tracker v6_02 event product, the saved 1k and 50k training lists, and the fixed companion validation and evaluation lists. Preparation and scoring run on CPU with the base package dependencies.

## Prepare once where the raw data are mounted

`PREPARED` is the companion campaign's `research_reports/transformer_downstream_20261104/preparation` directory. It must contain `data_ready.json` and the four frozen `momentum_{train_1000,train_50000,validation,evaluation}.npz` key/length files. `RAW` points at the original v6_02 RaggedMmap root and `OUT` is an external artifact directory.

```bash
clas12-benchmarks prepare-momentum --data "$RAW" --prepared "$PREPARED" --output "$OUT/tracks.npz"
clas12-benchmarks inspect-momentum "$OUT/tracks.npz"
```

Preparation checks the product metadata, ordered track keys, lengths and frozen raw feature/segmentation/regression content digests, then resolves each row to its original source-file/event identity. Validation and evaluation both come from the source product's test split and use distinct physical events. The 1k training list indexes the 50k list in its own frozen order. The output publishes atomically to a new filename.

The compressed manifest can be evaluated on a machine without the raw data. It contains one row per accepted track, offsets into packed xyz cluster centers in cm, and original within-event point indices. Truth is the mean of finite per-component entrance values for that segment, converted from MeV to GeV. CVT uses the first jointly finite row within the same segment for track p, track theta and trajectory entrance phi. Unavailable CVT vectors are kept as missing. The stored SHA-256 of the original product metadata and companion preparation receipt tie the pack to its sources. The evaluator also reports the SHA-256 of the pack it actually read.

```python
from clas12_benchmarks.momentum_manifest import TrackManifest

tracks = TrackManifest(manifest_path)
i = tracks.rows("train_1000")[0]
source_file, event, segment_label = tracks.identity(i)
xyz_cm, original_indices = tracks.hits(i)
target_gev = tracks.truth[i]
```

Use `hits` for measured inputs. Truth and CVT are separate arrays for supervision and evaluation. The supported role selections are `train_1000`, `train_50000`, `validation` and `evaluation`.

## Score a prediction file

The v1 interchange is CSV or gzip-compressed CSV. Write exactly one row per track in the chosen role. Row order is immaterial. The key is `source_file,event,segment_label`, including the full original source filename. For Cartesian fits add `pred_px_gev,pred_py_gev,pred_pz_gev`. For scalar transverse-momentum fits add `pred_pt_gev`. The evaluator rejects nonfinite predictions, unknown or duplicate identities and incomplete populations.

```bash
clas12-benchmarks evaluate-momentum "$OUT/tracks.npz" "$OUT/predictions.csv.gz" --role evaluation
```

JSON output separates scores for all tracks from paired model/CVT scores for the CVT-valid subset. Vector scores include component and magnitude MAE, transverse-momentum errors, signed bias, relative width and tails, and theta and wrapped phi errors. The relative width is half the distance between the 16th and 84th percentiles of signed relative magnitude error. The output names identify MeV, degrees or percent. Relative metrics count tracks with true momentum above 1e-12 GeV. Scalar predictions are scored in pT only.

## Replay a companion run

`EVAL` is a saved momentum run's evaluation directory. `TABLES` is the corresponding budget directory such as `comparison/momentum/1k_with_refinement` in the downstream comparison. Set `MODEL` to its table row, for example `mamba3`, `nope` or `adapteronly`.

```bash
clas12-benchmarks evaluate-momentum "$OUT/tracks.npz" "$EVAL/predictions.csv.gz" \
  --companion --reference-dir "$TABLES" --reference-model "$MODEL" \
  --reference-summary "$EVAL/summary.json" > "$OUT/replay.json"
```

The companion reader checks each prediction's original event row, segment, cluster count, truth and CVT vector against the manifest. It computes scores from manifest truth. Small numerical differences are expected because the companion reduced track truth in float32. The two CSV checks cover every score for the model on all tracks and for the model and CVT on common tracks. The summary check also compares native pT MAE and RMSE for both methods. Counts and tail fractions must agree to rounding. For continuous table values the check uses absolute tolerance 2e-5 in the reported unit and relative tolerance 2e-6. A mismatch exits with an error.

## Cluster HDF5 momentum

Start from a v2 event cohort prepared with `--reader hdf5`. This route freezes
tracks from that cohort without importing legacy raw-row keys:

```bash
clas12-benchmarks prepare-momentum-hdf5 --cohort "$OUT/cohort/cohort.json" \
  --output "$OUT/tracks-hdf5.npz" --min-clusters 12 --max-clusters 100 \
  --cvt-pass cvt --min-cvt-purity 1.0
clas12-benchmarks inspect-momentum "$OUT/tracks-hdf5.npz"
```

Compact primary segments resolve through positive `clusters.primary_tid` to
event-local `particles.tid`. Truth is the valid, finite observed direct-primary
vector already in GeV/c. It is neither the generated momentum nor the legacy
per-point finite mean. All three components must be valid; exclusion counts and
the selected multiplicity bounds are recorded. The default 12–100 cluster cut
is explicit and configurable. The pack preserves measured xyz and original
within-event point indices; model inputs remain separate from truth and CVT.

The reference requires one qualified, unique-owner native candidate in the
chosen pass, positive contributor overlap, the declared minimum purity, valid
benchmark p/theta, and a consistent same-pass entrance trajectory. Multiple
qualified candidates remain ambiguous. Missing or ambiguous CVT stays missing
without removing the model track. `cvtrec` can be chosen explicitly; no legacy
first-finite-row association is applied to native HDF5 tables.

V2 packs expose `train`, `validation`, and `evaluation` roles. Optional
`--train-track-budgets 1000 50000` adds nested prefixes `train_1000` and
`train_50000` if those counts fit the frozen training track population. These
are track budgets; the event cohort has its own event counts. Pack publication
is atomic and refuses to overwrite an existing filename.

V2 prediction CSV or gzip CSV uses these identity and contract columns:

```text
source_manifest_id,source_id,original_ordinal,primary_tid,schema,momentum_unit,manifest_sha256
```

Set `schema` to `clas12_momentum_predictions_v2`, `momentum_unit` to `GeV`, and
`manifest_sha256` to `TrackManifest(pack_path).sha256` on every row. Append either
`pred_px_gev,pred_py_gev,pred_pz_gev` or `pred_pt_gev`. Write one row per track in
the selected role; row order is immaterial. `TrackManifest.identity_fields` and
`identity(i)` supply the keys. Evaluate through the existing command:

```bash
clas12-benchmarks evaluate-momentum "$OUT/tracks-hdf5.npz" "$OUT/predictions.csv.gz" \
  --role evaluation
```

The evaluator dispatches by pack schema, rejects incompatible prediction
fingerprints/units and legacy companion rows for HDF5, and retains the existing
all-track/common-CVT metric definitions. Both formats remain readable without
the raw data; h5py is needed for HDF5 preparation, not packed scoring.

## CVT strip momentum

The standalone strip benchmark uses the complete `cvt-strips-h5-v1` product.
It predicts the published observed entrance momentum from truth-associated
strips on at least four distinct BST/BMT layers. The 307-feature MLP uses
charge-weighted layer geometry and readout summaries, approximate BST stereo
crossings, and pairs of transverse measurements for signed curvature estimates.
The trainable head has four residual blocks of width 320 and predicts log
momentum, local azimuth and pseudorapidity. Its outputs are converted to
Cartesian momentum in GeV/c for evaluation. The implementation uses PyTorch
layers and trains from scratch.

Use the `hdf5`, `tracking` and `plotting` extras with a CUDA PyTorch build
for training and checkpoint evaluation. The commands below reproduce the
large-cohort stage on `cvt_strip_hdf5_20261105_001`. Set `STRIPS` to that
product and `OUT` to an artifact directory outside the checkout.

```bash
python -m clas12_benchmarks.strip_momentum.cache_strips \
  --root "$STRIPS" --split train --events 1500000 --layer-only --out "$OUT/train"
python -m clas12_benchmarks.strip_momentum.cache_strips \
  --root "$STRIPS" --split validation --events 16000 --out "$OUT/validation"
python -m clas12_benchmarks.strip_momentum.cache_strips \
  --root "$STRIPS" --split test --events 60000 --out "$OUT/test"

mkdir -p "$OUT/fits" "$OUT/evaluation"
python -m clas12_benchmarks.strip_momentum.strip_adapter \
  --cache "$OUT/train" --val-cache "$OUT/validation" --out "$OUT/fits" \
  --train-size 2737603 --val-size 18000 --steps 45000 --batch 512 \
  --interval 4500 --amp \
  --arm helix:angular:0.0008:320:4:42 \
  --arm helix:angular-weighted:0.00065:320:4:23
python -m clas12_benchmarks.strip_momentum.evaluate_strips \
  --final "$OUT/fits" --train-cache "$OUT/train" \
  --validation "$OUT/validation" --test "$OUT/test" --out "$OUT/evaluation"
python -m clas12_benchmarks.strip_momentum.plot_residuals_vs_true_p \
  "$OUT/evaluation" "$OUT/plots"
```

The cache builder samples windows across the published splits and records the
dataset manifest hash and ordered track identities. The evaluator selects
checkpoints and ensemble size on validation momentum MAE. Test output includes
all eligible tracks and the subset with a unique, finite first-pass COATJAVA
match. The strip model receives truth groupings. COATJAVA uses its reconstructed
track memberships. The paired result measures momentum conditional on those inputs.

The November 18, 2026 a02c run measured 58.9 MeV/c momentum MAE and a 2.68%
central 68% half-width in relative error for the validation-selected single
model on 109,629 test tracks. The archived campaign includes smaller screening
stages and a CLAS12 Mamba2 reference. The standalone evaluator accepts
archived stage rankings through `--screen` and `--large` and replays the
standalone fits using their saved checkpoints. The saved Mamba2 reference
predictions can be scored from the original campaign's prediction archive.
