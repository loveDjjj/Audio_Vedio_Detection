from __future__ import annotations

import csv
import json
from pathlib import Path


def _is_real_real(record: dict) -> bool:
    return (
        record.get("label") == "real"
        and not bool(record.get("audio_fake", False))
        and not bool(record.get("video_fake", False))
    )


def _is_fake_fake(record: dict) -> bool:
    return (
        record.get("label") == "fake"
        and bool(record.get("audio_fake", False))
        and bool(record.get("video_fake", False))
    )


def build_real_fullfake_official_splits(records: list[dict]) -> dict[str, list[dict]]:
    split_map: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for record in records:
        split_name = str(record.get("split", "")).strip().lower()
        if split_name not in {"train", "validation", "test"}:
            continue
        if not (_is_real_real(record) or _is_fake_fake(record)):
            continue

        target_split = "val" if split_name == "validation" else split_name
        split_map[target_split].append(record)

    for split_name in split_map:
        split_map[split_name].sort(key=lambda item: str(item["video_path"]))
    return split_map


def to_csv_row(record: dict) -> dict:
    return {
        "relative_path": record["video_path"],
        "label": 0 if record["label"] == "real" else 1,
        "label_name": record["label"],
        "split": record["split"],
        "language": record["language"],
        "generative_method": record["generative_method"],
        "open_set_model": record["open_set_model"],
        "open_set_language": record["open_set_language"],
        "audio_generative_method": record.get("audio_generative_method", ""),
        "audio_fake": record.get("audio_fake", False),
        "video_fake": record.get("video_fake", False),
    }


def write_split_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "relative_path",
        "label",
        "label_name",
        "split",
        "language",
        "generative_method",
        "open_set_model",
        "open_set_language",
        "audio_generative_method",
        "audio_fake",
        "video_fake",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(to_csv_row(record))


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
