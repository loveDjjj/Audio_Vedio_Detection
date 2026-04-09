from __future__ import annotations

import torch
import torch.nn as nn

from src.models.avhubert_backbone import AVHubertVideoBackbone
from src.models.classifier_head import BinaryClassifierHead
from src.models.pooling import build_temporal_pooling


class AVHubertBinaryDetector(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        avhubert_repo,
        freeze_backbone: bool,
        pooling: str,
        classifier_dropout: float,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = AVHubertVideoBackbone(
            checkpoint_path=checkpoint_path,
            avhubert_repo=avhubert_repo,
            freeze=freeze_backbone,
        )
        self.pool = build_temporal_pooling(pooling)
        self.head = BinaryClassifierHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=hidden_dim,
            dropout=classifier_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, video: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        features, feature_padding_mask = self.backbone(video=video, padding_mask=padding_mask)
        pooled = self.pool(features, feature_padding_mask)
        return self.head(pooled)
