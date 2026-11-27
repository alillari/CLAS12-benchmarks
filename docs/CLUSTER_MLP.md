# CVT cluster MLP for entrance momentum

This external baseline estimates laboratory `[px, py, pz]` in GeV/c from
measured clusters assigned to a single track. The HDF5 benchmark uses
truth-associated cluster membership and observed direct-primary momentum at
the first-layer/Hilbert reference. Track finding and real-data transfer require
separate evaluations.

## Recipes from the November 25, 2026 study

| Recipe | Training labels | Updates × batch | Momentum magnitude MAE (MeV/c) | Δp/p central 68% half-width (%) | θ MAE (°) | Wrapped φ MAE (°) |
|---|---:|---:|---:|---:|---:|---:|
| Aligned 24, `g18_endz` | 1,872,952 | 45,000 × 512 | 35.19 | 2.61 | 0.412 | 0.113 |
| Compact full-label 16, `bmt_cslope_outer16` | 1,872,952 | 45,000 × 512 | 36.60 | 2.68 | 0.425 | 0.115 |
| Engineered 10k-label 17, `bst_angle17` | 10,000 | 6,500 × 256 | 52.53 | 3.52 | 0.838 | 0.242 |
| Distinct 218, `distinct218` | 1,872,952 | 45,000 × 512 | 35.37 | 2.60 | 0.391 | 0.114 |
| Distinct 218, `distinct218` | 10,000 | 6,500 × 256 | 54.09 | 3.60 | 0.748 | 0.239 |

The entries average metrics from seeds 23, 42 and 91 on the same 74,789
held-out simulation tracks. The upstream cohort eligibility join was retained.
The roster used four or more occupied layers, observed `0 < p < 3 GeV/c`, and
`35° ≤ θ ≤ 125°`. Each test track was a positive hadron. The central half-width
is half the distance between the 16th and 84th percentiles of signed Δp/p.

1. **Aligned 24 is the default benchmark.** Both endpoints in all six BST
   layers contribute local y. Three BMT-Z centers contribute local y, three
   BMT-C centers contribute z, and the six second BST endpoints contribute z.
   The original 307-input model scored 35.25 MeV/c and the 218 distinct inputs
   scored 35.37 MeV/c at the same label and update budget.
2. **Compact 16 preserves most aggregate performance.** It keeps the 12 BST
   endpoint y values, replaces the six BMT measurements with fitted BMT-Z y and
   BMT-C z at 18 cm plus the BMT-C dz/dr, and keeps only the outermost second
   BST endpoint z. It was the smallest tested full-label recipe to meet the
   study's joint magnitude and angular tolerances. On the 4,605 test tracks
   with 4–6 occupied layers, the central 95% θ half-width was 6.20° versus
   4.67° for aligned 24.
3. **Engineered 17 uses scarce labels well.** It keeps the 12 BST endpoint y
   values and adds three estimates from a chord and longitudinal fit (log p
   proxy, η and entrance tangent φ), followed by fitted BMT-C z at 18 cm and
   dz/dr. At 10k labels it improved magnitude MAE from 56.70 to 52.53 MeV/c
   relative to aligned 24. The distinct 218 at that budget scored 54.09 MeV/c
   and had lower θ MAE, 0.748°. The two label budgets yield different rankings.

The distinct 218 recipe preserves the study's ordered selection of varying,
nonduplicate columns from the original 307 on the audited cohort. It includes
local layer coordinates, laboratory x and y, pitch, occupancy, BMT-C sweep, BST
stereo estimates and transverse and longitudinal fit summaries.

## Matched 108-coordinate controls

Each recipe uses the same 108 center and endpoint coordinates in BST 1–6
followed by BMT 1–6. All five results use the full-label budget in the table
above and average the three seeds on the same held-out tracks.

| Recipe | Input frame | φ target frame | φ readout | Momentum MAE (MeV/c) |
|---|---|---|---|---:|
| `xyz108_local` | Aligned | Local | Scalar | 34.98 |
| `xyz108_lab_local_target` | Laboratory | Local | Scalar | 70.43 |
| `xyz108_lab` | Laboratory | Laboratory | Scalar | 96.95 |
| `xyz108_local_circle` | Aligned | Local | Circular | 34.96 |
| `xyz108_lab_circle` | Laboratory | Laboratory | Circular | 69.72 |

Local φ targets use the measured α during training and decode by adding it
back. Circular heads derive φ from a pair of FP32 outputs. These results
apply to the tested training recipe and simulation. For comparisons against
the frozen Campaign 4 M6 fit, use the separate matched 100k-label
result (aligned 24 at 39.20 MeV/c, frozen M6 at 50.66 MeV/c). Both used 45,000
updates at batch 128 and the same test tracks.

## Train and score a frozen HDF5 cohort

Set `DATA_ROOT` to the `cvt-clusters-h5-v1` publication root containing
`manifest.json` and `OUT` to a writable artifact directory. From the
repository root, install the benchmark and prepare a frozen cohort:

```bash
python -m pip install -e '.[hdf5,tracking]'
python -m clas12_benchmarks.prepare_cohort "$DATA_ROOT" "$OUT/cohort" \
  --reader hdf5 --train 10000 --val 1000 --test 2000 --seed 1337
COHORT="$OUT/cohort/cohort.json"
```

If you already have a frozen cohort, set `COHORT` to its `cohort.json` and
create `OUT` before running the steps below.

```bash
clas12-benchmarks prepare-momentum-hdf5 --cohort "$COHORT" \
  --output "$OUT/tracks.npz" --min-clusters 4 --cluster-mlp-geometry
python -m clas12_benchmarks.cluster_mlp train --manifest "$OUT/tracks.npz" \
  --arm g18_endz --seed 23 --output "$OUT/aligned24.pt"
python -m clas12_benchmarks.cluster_mlp predict --manifest "$OUT/tracks.npz" \
  --checkpoint "$OUT/aligned24.pt" --output "$OUT/aligned24.csv.gz"
clas12-benchmarks evaluate-momentum "$OUT/tracks.npz" "$OUT/aligned24.csv.gz"
```

The geometry option reads detector, layer, endpoints, pitch and length from
the HDF5 product. It requires four occupied layers and positive momentum magnitude in
addition to the regular cluster-count and truth-validity cuts. Inspect the
pack to see the resulting role populations, or read `selection_counts` in its
metadata for the per-role cut flow. The study's phase-space cuts and frozen
eligibility roster apply to its reported numbers. Use the archived study cache
for that population.

For each track, the model uses equal-weight cluster means in BST 1–6 followed
by BMT 1–6. The beam defines z. The sum of occupied BST and BMT-Z transverse
centers defines azimuth α, with fallback to all centers if those layers are
absent. Geometry rotates by −α and the decoder adds α back. Coordinates use
30 cm as one unit and local y gets a factor of eight. Empty slots are zero.
Each BMT fit weights occupied layers equally. It returns the observed value
and zero slope with one layer, or zeros with none.

The model fits standardized log p, wrapped local φ and η with the study's
momentum-weighted smooth L1 loss and four 320-wide residual blocks. It
standardizes on the selected training prefix and selects weights on validation
momentum magnitude MAE. The default is the full-label budget for every
recipe except `bst_angle17`, which defaults to its 10k-label budget.
Pass any recipe name from the tables via `--arm`. `--labels`, `--steps`,
and `--batch` override the budget. To rerun the 10k-label 218 control, pass
`--arm distinct218 --labels 10000 --steps 6500 --batch 256`. `--device cpu`
uses FP32.
CUDA uses BF16 autocast where available, with `--fp32` to disable it.
For a quick check, add `--device cpu --steps 2 --batch 2` to the default
aligned-24 training command. The same prediction and evaluation commands
will run with the resulting checkpoint. Training and validation need at
least two eligible tracks each, and evaluation needs at least one.

```bash
python -m unittest discover -s tests -p test_cluster_mlp.py -v
```

## Replay the archived study

The November 25 `cluster-mlp-feature-study.zip` research archive contains
`REPRODUCE.txt`, the original scripts, the frozen roster and field digests in
`source/replica/`, the held-out comparison in `results/confirmation.json`, the
frame control in `diagnostics/lab108/xyz_summary.json` and the distinct-column
comparison in `diagnostics/reference/`. Their receipts identify the managed
jobs that retain the larger checkpoints. `models/checkpoints.json` indexes
the supplied compact checkpoints and their SHA-256 digests. The archive pins the
`cvt_cluster_hdf5_20261103_002` manifest to SHA-256
`f813cad285b0c122764e027062fecdb5c90f3ae87dd4fc119e3300d51fff45c4`.
The archive's `source/cohort.py --replica` mode reconstructs the exact roster
from the cluster HDF5. Its training arrays can be used directly with the port.
Set `STUDY` to the extracted `cluster_feature_study` directory. The bundled
64-track sample exercises checkpoint inference without rebuilding the cache.

```bash
python -m clas12_benchmarks.cluster_mlp predict-geometry \
  --checkpoint "$STUDY/models/full-label-24-native.pt" \
  --geometry "$STUDY/models/example_geometry.npy" \
  --output "$OUT/aligned24-example.npy" --device cpu
```

Set `CD_CACHE` to the rebuilt cache directory for training or held-out replay.

```bash
python -m clas12_benchmarks.cluster_mlp train --study-cache "$CD_CACHE" \
  --arm bst_angle17 --seed 23 --output "$OUT/engineered17.pt"
python -m clas12_benchmarks.cluster_mlp predict-geometry \
  --checkpoint "$STUDY/models/full-label-24-native.pt" \
  --geometry "$CD_CACHE/test_cluster.npy" --output "$OUT/aligned24.npy"
```

`predict-geometry` accepts the archived float32 `[N,12,19]` layout or the
current pack's `[N,12,12]` layout. The latter stores center xyz, two endpoint
xyz triples, log1p cluster count, mean BMT-C sweep and mean log1p(100 × pitch
in cm). Older `[N,12,10]` geometry omits the final two fields and supports
all recipes except `distinct218`. Output rows preserve input order. In the study cache they
align with `test_truth.npy` and `test_id.npy`. The archived weight files for
the other recipes are `full-label-16.pt` and `10k-label-17.pt`. A fit with
the port uses eager updates. The archived results used CUDA graph replay and
are the reference for the table above.
