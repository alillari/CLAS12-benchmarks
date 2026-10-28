# Porting status and decisions

## Provenance

The initial source audit inspected the complete public FM4NPP repository tree at
`863cb28ad798d4753569737563a754cb2a0e6284` and the local
`David_code/home/shuhang/FM4NPP/FM4NPP` downstream archive. Neither supplied these
external-method adaptations. The user confirmed no other known code source.

Reference: [FM4NPP arXiv v1, comparative methods in Appendix C](https://arxiv.org/html/2508.14087v1).
The paper describes settings but does not pin the external source revisions.
Imported projects' current pinned revisions are starting points, not established
publication-era code versions. The source lock and reference spec retain this distinction.

## EggNet

The upstream implementation is retained unchanged. Our adapter produces its named
per-hit feature columns, full-event point identities and training truth edges.
CLAS12 signal label zero is shifted to upstream particle ID one; background -1 is
mapped to zero. Directed positive pairs include every distinct same-track pair;
noise is never treated as one true track. A configurable edge-count limit fails
instead of silently changing supervision. Model neighbors use coordinates only.

The twelve input features concatenate each point's normalized coordinates with
those of its two distinct nearest neighbors in four-coordinate space. For exact
distance ties, SciPy's neighbor choice is used; no cross-library tie equivalence
is claimed. The exporter preserves event and point order and refuses overwrite.

Remaining work: finalize architecture flags not specified in the FM4NPP paper,
choose and validate positive/negative pair sampling and loss reductions, integrate
CLAS12 validation metrics and DBSCAN inference, test the upstream GPU runtime and
strict checkpoint/resume behavior, then run a real-data smoke. Upstream defaults
must not be represented as the missing FM4NPP settings.

## Exa.TrkX

The imported `exatrkx-iml2020` publication pipeline includes embedding, filtering,
TensorFlow Interaction Network graph code, and a PyTorch GNN option. The earlier
CTD pipeline was inspected but is not imported. We have not established which
revision or framework the FM4NPP authors used. FM4NPP's all-same-track edge labels
differ from immediate-neighbor truth in upstream preprocessing. Coordinate
conversion alone is insufficient to call this adapted.

Remaining work: replace TrackML feature/truth preparation, reconcile removal of
cell features, implement all-same-track supervision and validate each learned
stage. Recalibrate thresholds using validation only; the paper's TPC values are
reference values, not approved CLAS12 cuts. Validate graph-to-track clustering
without truth-assisted relabeling and test the full pipeline in an isolated runtime.

## Semantic graph networks

Our reconstruction uses the four actual PyG convolution classes, six graph layers,
a two-hidden-layer 256-wide encoder, and classifier hidden widths 128/64. It
returns logits for cross entropy. Directed geometry neighbors are constructed
per event before batching; there are no cross-event edges.

Choices not resolved by the paper: ReLU activations; 256-wide graph state; an
explicit encoder output projection; one GAT head; no dropout, batch normalization
or added residual connections; PyG 2.4.0 defaults for each convolution, including
operator-specific self loops and aggregation. The helper does not add self edges;
some convolution operators do. These choices are recorded, not asserted to match
the authors' unavailable code. CPU tests check all four operators' gradient updates
and disk checkpoint reload. There is no production training runner yet.

Remaining work: define PID/noise label maps from dataset metadata, ignore-label
handling, common training/validation selection and budget, and a reusable semantic
training runner; validate on real data. Momentum regression needs a new task head
and must be identified as our extension rather than an original FM4NPP baseline.

## OneFormer3D

The complete original project and S3DIS configuration are retained unchanged,
including the original license. Its unmodified example is an indoor segmentation
pipeline with color features, class definitions and augmentations that do not
directly describe CLAS12. We have not supplied a misleading runnable renamed config.

Remaining work: implement event-level sparse coordinates and superpoint-to-hit
mapping, replace feature/class handling and augmentations, retain the U-Net and
query decoder while adapting semantic-only loss/prediction, and validate the
sparse-convolution environment. The paper's TPC voxel-grid dimensions require an
explicit CLAS12 geometry decision. Track finding or momentum heads are extensions.

## Shared evaluation

The manifest freezes source-file/event identities across all roles. Targets never
enter geometry neighbor construction. Prediction exports reject missing or duplicate
point identities and restore source order. Unknown/invalid values are failures,
not reasons to reduce the evaluation population. Connecting this interchange to
the existing CLAS12 evaluator is separate pending work; neither current metric
equivalence nor agreement with published scores has been established.
