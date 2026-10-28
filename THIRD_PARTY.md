# Third-party sources

| Source | Directory | Upstream license |
|---|---|---|
| [EggNet](https://github.com/exatrkx/EggNet) | `vendor/eggnet` | Apache-2.0; original LICENSE retained |
| [Exa.TrkX publication pipeline](https://github.com/exatrkx/exatrkx-iml2020) | `vendor/exatrkx` | Apache-2.0; original LICENSE retained |
| [OneFormer3D](https://github.com/filaPro/oneformer3d) | `vendor/oneformer3d` | CC BY-NC 4.0; original LICENSE retained |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | Installed optional dependency | MIT; dependency license applies |

Source commit identities are in `sources.lock.json`; upstream sources were imported
without local modifications. Our wrappers and paper-based graph reconstruction
live outside the imported trees. There is no blanket license grant for third-party
code. In particular, OneFormer3D's upstream noncommercial terms remain applicable.

Method references and bibliography are available in each upstream README and the
[FM4NPP paper](https://arxiv.org/abs/2508.14087). Use method-specific citations when
reporting benchmark results. This repository is an independent CLAS12 adaptation,
not an official distribution of the FM4NPP authors' external baseline adaptations.
