from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if padding_mask is None:
            return features.mean(dim=1)

        valid_mask = (~padding_mask).unsqueeze(-1)
        masked_features = features * valid_mask
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)
        return masked_features.sum(dim=1) / valid_counts


def build_temporal_pooling(name: str) -> nn.Module:
    if name != "masked_mean":
        raise ValueError(f"Unsupported pooling strategy: {name}")
    return MaskedMeanPooling()
