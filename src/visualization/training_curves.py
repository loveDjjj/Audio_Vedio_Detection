from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_training_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_curve_series(history: list[dict]) -> dict:
    epochs = [int(item["epoch"]) for item in history]
    metric_names = sorted(history[0]["train"].keys())

    train_series = {metric: [float(item["train"][metric]) for item in history] for metric in metric_names}
    val_series = {metric: [float(item["val"][metric]) for item in history] for metric in metric_names}
    return {
        "epochs": epochs,
        "train": train_series,
        "val": val_series,
    }


def plot_training_summary(summary_path: Path, output_path: Path) -> Path:
    summary = load_training_summary(summary_path)
    history = summary.get("history", [])
    if not history:
        raise ValueError(f"Summary has no history: {summary_path}")

    series = build_curve_series(history)
    epochs = series["epochs"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metric_groups = [
        ("loss", ["loss"]),
        ("accuracy", ["accuracy"]),
        ("f1", ["f1"]),
        ("precision_recall", ["precision", "recall"]),
    ]

    for axis, (title, metrics) in zip(axes, metric_groups):
        for metric in metrics:
            axis.plot(epochs, series["train"][metric], marker="o", label=f"train_{metric}")
            axis.plot(epochs, series["val"][metric], marker="s", label=f"val_{metric}")
        axis.set_title(title.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()

    fig.suptitle(summary_path.parent.name)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
