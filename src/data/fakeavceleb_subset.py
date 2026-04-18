from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FakeAVCelebRecord:
    relative_path: str
    label: int
    label_name: str
    source: str
    target1: str
    target2: str
    method: str
    category: str
    type: str
    race: str
    gender: str
    filename: str


SUPPORTED_TYPES = {
    "RealVideo-RealAudio": (0, "real"),
    "FakeVideo-FakeAudio": (1, "fake"),
}


def _normalize_relative_dir(dirpath: str) -> Path:
    path = Path(dirpath.strip())
    parts = list(path.parts)
    if parts and parts[0] == "FakeAVCeleb":
        parts = parts[1:]
    return Path(*parts)


def _parse_row(row: list[str], root: Path) -> FakeAVCelebRecord | None:
    if len(row) < 10:
        raise ValueError(f"Expected at least 10 columns in FakeAVCeleb metadata, got {len(row)}: {row}")

    source, target1, target2, method, category, type_name, race, gender, filename, dirpath = [
        item.strip() for item in row[:10]
    ]
    if type_name not in SUPPORTED_TYPES:
        return None

    relative_path = (_normalize_relative_dir(dirpath) / filename).as_posix()
    full_path = root / relative_path
    if not full_path.exists():
        return None

    label, label_name = SUPPORTED_TYPES[type_name]
    return FakeAVCelebRecord(
        relative_path=relative_path,
        label=label,
        label_name=label_name,
        source=source,
        target1=target1,
        target2=target2,
        method=method,
        category=category,
        type=type_name,
        race=race,
        gender=gender,
        filename=filename,
    )


def load_fakeavceleb_records(root: Path) -> list[FakeAVCelebRecord]:
    metadata_path = root / "meta_data.csv"
    records: list[FakeAVCelebRecord] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            record = _parse_row(row, root)
            if record is not None:
                records.append(record)
    records.sort(key=lambda item: item.relative_path)
    return records


def select_real_fullfake_records(records: list[FakeAVCelebRecord]) -> list[FakeAVCelebRecord]:
    return sorted(records, key=lambda item: item.relative_path)


def _allocate_counts(total: int, ratios: tuple[float, ...]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(raw) for raw in raw_counts]
    remainder = total - sum(counts)
    ordering = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in ordering[:remainder]:
        counts[index] += 1
    return counts


def split_records(records: list[FakeAVCelebRecord], seed: int) -> dict[str, list[FakeAVCelebRecord]]:
    grouped: dict[tuple[str, str], list[FakeAVCelebRecord]] = defaultdict(list)
    for record in records:
        stratum_key = (record.label_name, "real" if record.label == 0 else record.method)
        grouped[stratum_key].append(record)

    split_names = ("train", "val", "test")
    target_totals = _allocate_counts(len(records), (0.8, 0.1, 0.1))
    current_totals = [0, 0, 0]
    split_map: dict[str, list[FakeAVCelebRecord]] = {"train": [], "val": [], "test": []}
    for (label_name, method), items in sorted(grouped.items()):
        shuffled = list(items)
        random.Random(f"{seed}:{label_name}:{method}").shuffle(shuffled)
        raw_counts = [len(shuffled) * ratio for ratio in (0.8, 0.1, 0.1)]
        group_counts = [math.floor(raw) for raw in raw_counts]
        remaining = len(shuffled) - sum(group_counts)
        while remaining > 0:
            ordering = sorted(
                range(len(split_names)),
                key=lambda index: (
                    raw_counts[index] - group_counts[index],
                    target_totals[index] - (current_totals[index] + group_counts[index]),
                    -index,
                ),
                reverse=True,
            )
            group_counts[ordering[0]] += 1
            remaining -= 1

        start = 0
        for index, split_name in enumerate(split_names):
            end = start + group_counts[index]
            split_map[split_name].extend(shuffled[start:end])
            current_totals[index] += group_counts[index]
            start = end

    return {split_name: sorted(split_rows, key=lambda item: item.relative_path) for split_name, split_rows in split_map.items()}


def build_split_rows(records: list[FakeAVCelebRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def write_split_csv(path: Path, records: list[FakeAVCelebRecord]) -> None:
    fieldnames = [
        "relative_path",
        "label",
        "label_name",
        "source",
        "target1",
        "target2",
        "method",
        "category",
        "type",
        "race",
        "gender",
        "filename",
    ]
    rows = build_split_rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    split_map: dict[str, list[FakeAVCelebRecord]],
    source_records: list[FakeAVCelebRecord],
    selected_records: list[FakeAVCelebRecord],
    seed: int,
) -> dict:
    return {
        "dataset": "FakeAVCeleb",
        "selection": ["RealVideo-RealAudio", "FakeVideo-FakeAudio"],
        "split_strategy": "stratified_random_video_level",
        "seed": seed,
        "source_videos": len(source_records),
        "selected_videos": len(selected_records),
        "source_labels": dict(Counter(record.label_name for record in source_records)),
        "source_fake_methods": dict(Counter(record.method for record in source_records if record.label == 1)),
        "selected_labels": dict(Counter(record.label_name for record in selected_records)),
        "selected_fake_methods": dict(Counter(record.method for record in selected_records if record.label == 1)),
        "splits": {
            split_name: {
                "videos": len(split_rows),
                "labels": dict(Counter(record.label_name for record in split_rows)),
                "methods": dict(Counter(record.method for record in split_rows if record.label == 1)),
            }
            for split_name, split_rows in split_map.items()
        },
    }


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
