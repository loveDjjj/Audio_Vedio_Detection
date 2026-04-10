from __future__ import annotations

import torch
import torch.nn as nn

from src.models.avhubert_backbone import AVHubertBackbone


def _masked_logsumexp(frame_logits: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return torch.logsumexp(frame_logits, dim=1)

    masked_logits = frame_logits.masked_fill(padding_mask, torch.finfo(frame_logits.dtype).min)
    video_logits = torch.logsumexp(masked_logits, dim=1)
    invalid_rows = padding_mask.all(dim=1)
    if invalid_rows.any():
        video_logits = video_logits.masked_fill(invalid_rows, 0.0)
    return video_logits


class SSRDFDAVHubertLinearProbe(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 1024) -> None:
        super().__init__()
        self.backbone = backbone

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.head = nn.Linear(feat_dim, 1)

    def forward(
        self,
        audio: torch.Tensor | None,
        video: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_padding_mask = self.backbone(
            audio=audio,
            video=video,
            padding_mask=padding_mask,
        )
        frame_logits = self.head(features).squeeze(-1)
        video_logits = _masked_logsumexp(frame_logits, feature_padding_mask)
        return video_logits, frame_logits


class AVHubertBinaryDetector(SSRDFDAVHubertLinearProbe):
    def __init__(
        self,
        checkpoint_path,
        avhubert_repo,
        freeze_backbone: bool,
        feat_dim: int | None = None,
    ) -> None:
        backbone = AVHubertBackbone(
            checkpoint_path=checkpoint_path,
            avhubert_repo=avhubert_repo,
            freeze=freeze_backbone,
        )
        resolved_feat_dim = backbone.output_dim if feat_dim is None else feat_dim
        if resolved_feat_dim != backbone.output_dim:
            raise ValueError(
                f"Configured feat_dim={resolved_feat_dim} does not match AV-HuBERT output_dim={backbone.output_dim}."
            )
        super().__init__(backbone=backbone, feat_dim=resolved_feat_dim)
