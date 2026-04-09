from __future__ import annotations

import torch


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).long()
    targets = targets.long()

    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (targets == 0)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())

    total = max(len(targets), 1)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "tn": float(true_negative),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }
