# Momentum track manifest and evaluator

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
