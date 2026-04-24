from __future__ import annotations

import csv
import json
from collections import Counter
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


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _label_counts(records: list[dict]) -> dict[str, int]:
    return _sorted_counter(Counter(str(record["label"]) for record in records))


def _split_counts(records: list[dict]) -> dict[str, int]:
    counter: Counter = Counter()
    for record in records:
        split_name = str(record.get("split", "")).strip().lower()
        counter["val" if split_name == "validation" else split_name] += 1
    return _sorted_counter(counter)


def build_local_available_real_fullfake_official_splits(
    records: list[dict],
    raw_video_root: Path,
) -> tuple[dict[str, list[dict]], dict]:
    full_split_map = build_real_fullfake_official_splits(records)
    available_split_map: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    available_records: list[dict] = []
    missing_records: list[dict] = []

    for split_name, split_records in full_split_map.items():
        for record in split_records:
            video_path = raw_video_root / str(record["video_path"])
            if video_path.is_file():
                available_split_map[split_name].append(record)
                available_records.append(record)
            else:
                missing_records.append(record)

    requested_records = [record for split_records in full_split_map.values() for record in split_records]
    availability = {
        "raw_video_root": str(raw_video_root),
        "requested_videos": len(requested_records),
        "available_videos": len(available_records),
        "missing_videos": len(missing_records),
        "available_splits": {split_name: len(available_split_map[split_name]) for split_name in ("train", "val", "test")},
        "missing_splits": _split_counts(missing_records),
        "available_labels": _label_counts(available_records),
        "missing_labels": _label_counts(missing_records),
    }
    return available_split_map, availability


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
