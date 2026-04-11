from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def sample_records_by_generator(records: list[dict], sample_ratio: float, seed: int) -> list[dict]:
    if not (0 < sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["generative_method"])].append(record)

    sampled: list[dict] = []
    for generator, items in sorted(grouped.items()):
        shuffled = list(items)
        random.Random(f"{seed}:{generator}").shuffle(shuffled)
        keep = max(1, math.ceil(len(shuffled) * sample_ratio))
        sampled.extend(shuffled[:keep])
    sampled.sort(key=lambda item: str(item.get("video_path", item.get("relative_path", ""))))
    return sampled


def build_english_subset_splits(records: list[dict], train_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict]]:
    english_records = [record for record in records if record["language"] == "english"]
    train_records = [record for record in english_records if record["split"] == "train"]
    val_records = [record for record in english_records if record["split"] == "validation"]
    test_records = [record for record in english_records if record["split"] == "test" and record["open_set_model"]]

    return {
        "train": sample_records_by_generator(train_records, sample_ratio=train_ratio, seed=seed),
        "val": sorted(val_records, key=lambda item: str(item["video_path"])),
        "test": sample_records_by_generator(test_records, sample_ratio=test_ratio, seed=seed + 1),
    }


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
