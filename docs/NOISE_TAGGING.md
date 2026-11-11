# HDF5 noise tagging

The runner trains GATConv, GCNConv, GraphConv and SAGEConv on the exact event
lists exported by a CLAS12 FM noise adapter. Use `cohorts.json` and the
`data_contract.plan_id` from that adapter evaluation. The runner verifies the
publication, event identities and split membership, and fingerprints the selected
HDF5 content. All four models use the same imported cohorts and geometry graph.

```bash
python -m pip install -e '.[graphs,hdf5,evaluation]'
python -m clas12_benchmarks.noise_tagging \
  --data-root "$CVT_CLUSTER_HDF5" \
  --fm-cohorts "$FM_EVAL/cohorts.json" \
  --plan-id "$FM_PLAN_ID" \
  --output "$RUN_ROOT/noise-gnn-smoke" \
  --steps 2 --val-interval 1 --batch-size 8 --seed 71 \
  --class-balance inverse_frequency --device cpu
```

Omit `--operators` to run all four or pass their names to select a subset. Use
`--device cuda:0` with a compatible Torch/PyG installation. The runner uses AdamW,
cross-entropy, and a fixed update budget. Inverse-frequency weighting counts only
training clusters. The position-only features and graph normalization are fitted
or defined from the training data as in the external graph reconstruction.

The binary target is `primary_unassigned_or_lt4_is_noise_v2`. A cluster is noise
if it is unassigned or belongs to an MC segment with fewer than four owned clusters
in its complete event. All clusters remain in the graph. The COATJAVA comparison
flags the same clusters by `reco_segment < 0`. When importing cohorts from an older
FM evaluation, compare scores after applying the same truth definition to both.
Empty events remain in the cohort. A batch with zero clusters consumes no optimizer
update. The validation cohort needs signal clusters to calibrate retention.

The selected checkpoint minimizes pooled validation cross-entropy, with the
earliest checkpoint retained on ties. After reloading it, the runner reports the
FM default threshold of 0.5, the validation thresholds for 95% and 99% signal
retention and matched COATJAVA retention, and the campaign's exact maximum
validation balanced-accuracy point when both classes are present. It applies these
thresholds unchanged to test scores. Noise is predicted when `P(noise) > threshold`.

Each operator writes its checkpoint, validation history, split metrics, event
confusion records, and score archives with offsets into the imported event order.
The `adapter` key stores the graph model's metrics under the names used by the FM
evaluator. The shared `provenance.json` records source hashes and the runtime
environment. From a Git checkout it also captures the commit and tracked source
diff. A completed run has a top-level `summary.json` listing each result and the
shared cohort fingerprint. Use a fresh output directory for a new run.
