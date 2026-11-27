"""Matched-cohort controls using the native FM4NPP noise attention head."""
from pathlib import Path

import torch
from torch import nn

from .data import file_sha256

# Existing FM4NPP position-only dataset constants, in eta, phi, cm.
MINIMUM = (-2.5, -torch.pi, 6.)
SPAN = (4., 2 * torch.pi, 17.)


class FMNoiseClassifier(nn.Module):
    """Apply the native attention adapter, with optional frozen M4 final states."""
    def __init__(self, scales, checkpoint=None, sha256=None, cached=False):
        super().__init__()
        from fm4npp.models.mamba_config import build_backbone
        from fm4npp.pretrained import checkpoint_config
        from train.downstream import model as native

        self.scales = scales
        self.cached = cached
        self.backbone = None
        self.source = file_sha256(Path(native.__file__))
        if checkpoint is not None:
            if not sha256 or file_sha256(checkpoint) != sha256:
                raise ValueError("Frozen FM control checkpoint failed its SHA256 check")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = checkpoint_config(saved)
            expected = dict(backbone_version="mamba3", embed_dim=512,
                            num_layers_backbone=12, hilbert_angular_grid="unit_v2",
                            backbone_output_dim=4)
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError("Frozen FM control has an incompatible checkpoint contract")
            # The verified historical descriptor supplies fields omitted by the checkpoint.
            inputs = dict(klen=1, embed_method="pos_only", pe_method="nerf",
                          use_sincos_target=True, dropout=0.)
            self.backbone = build_backbone({**config, **inputs})
            self.backbone.load_state_dict({k.removeprefix("module."): v
                                           for k, v in saved["model_state"].items()}, strict=True)
            self.backbone.requires_grad_(False).eval()
        frozen = self.backbone is not None or self.cached
        self.head = native.AttentionHead(input_dim=512, num_layers=1, num_heads=4, num_output_dim=2,
                                         num_embedder_layers=0, num_feature_layers=1 if frozen else 12,
                                         embed_method="pos_only", pe_method="nerf")
        # The input-mode-specific branches exist in the native head. Exclude their
        # unused parameters from optimization and counts of active trainable weights.
        self.head.weighted_avg_weights.requires_grad_(False)
        if frozen:
            self.head.embedder.requires_grad_(False)
            self.head.embedder_norm.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def serialize(self, x, mask):
        from fm4npp.hilbert import assign_clas12_layer, clas12_hilbert_cells, encode

        eta, cosine, sine, radius = x.unbind(dim=-1)
        values = torch.stack((eta * self.scales["eta"], torch.atan2(sine, cosine),
                              radius * self.scales["radius"]), dim=-1)
        values = (values - x.new_tensor(MINIMUM)) / x.new_tensor(SPAN)
        radius_order = values[..., 2].masked_fill(~mask, torch.inf).argsort(dim=1, stable=True)
        ordered = values.gather(1, radius_order.unsqueeze(-1).expand_as(values))
        valid = mask.gather(1, radius_order)
        cells = clas12_hilbert_cells(ordered[..., 1].flatten(), ordered[..., 0].flatten())
        codes = encode(cells, num_dims=2, num_bits=10)
        bands = assign_clas12_layer(ordered[..., 2].flatten())
        key = ((bands.long() << 48) + codes.long()).reshape_as(mask)
        order = key.masked_fill(~valid, torch.iinfo(torch.long).max).argsort(dim=1, stable=True)
        ordered = ordered.gather(1, order.unsqueeze(-1).expand_as(ordered))
        valid = valid.gather(1, order)
        return ordered.masked_fill(~valid.unsqueeze(-1), -100.), valid, radius_order.gather(1, order)

    @torch.no_grad()
    def extract_features(self, x, mask):
        ordered, valid, order = self.serialize(x, mask)
        _, hidden, _ = self.backbone(ordered, return_z=True)
        values = self.backbone.norm(hidden[-1]).masked_fill(~valid.unsqueeze(-1), 0.)
        return torch.zeros_like(values).scatter(1, order.unsqueeze(-1).expand_as(values), values)

    def forward(self, x, mask):
        if self.cached:
            # The attention head is equivariant to the row order of its frozen features.
            ordered, valid, order = x, mask, None
            levels = x.unsqueeze(0)
        else:
            ordered, valid, order = self.serialize(x, mask)
            levels = None
            if self.backbone is not None:
                with torch.no_grad():
                    _, hidden, _ = self.backbone(ordered, return_z=True)
                    levels = self.backbone.norm(hidden[-1]).unsqueeze(0)
        # Empty lanes are masked after attention. Supply one harmless key so
        # softmax sees a finite entry even in an empty lane.
        safe = valid.clone()
        safe[:, 0] = True
        logits = self.head(ordered, feature=levels, pretrain=levels is not None,
                           padding_mask=safe)["pred_logits"]
        logits = logits.masked_fill(~valid.unsqueeze(-1), 0.)
        if order is None:
            return logits
        return torch.zeros_like(logits).scatter(1, order.unsqueeze(-1).expand_as(logits), logits)
