from __future__ import annotations

import torch.nn as nn


class BinaryClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]

        if hidden_dim > 0:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, 1),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, pooled_features):
        return self.network(pooled_features).squeeze(-1)
