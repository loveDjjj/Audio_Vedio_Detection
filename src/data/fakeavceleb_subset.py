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


def _sample_fake_records_by_method(
    fake_records: list[FakeAVCelebRecord], target_count: int, seed: int
) -> list[FakeAVCelebRecord]:
    if target_count < 1:
        return []
    if len(fake_records) < target_count:
        raise ValueError(
            f"Not enough fake FakeAVCeleb records to sample target_count={target_count}, got {len(fake_records)}"
        )

    grouped: dict[str, list[FakeAVCelebRecord]] = defaultdict(list)
    for record in fake_records:
        grouped[record.method].append(record)

    total = len(fake_records)
    selected: list[FakeAVCelebRecord] = []
    selected_paths: set[str] = set()
    remainders: list[tuple[float, str]] = []

    for method, items in sorted(grouped.items()):
        shuffled = list(items)
        random.Random(f"{seed}:{method}").shuffle(shuffled)
        raw_keep = target_count * len(items) / total
        keep = min(len(items), math.floor(raw_keep))
        chosen = sorted(shuffled[:keep], key=lambda item: item.relative_path)
        selected.extend(chosen)
        selected_paths.update(item.relative_path for item in chosen)
        remainders.append((raw_keep - keep, method))

    while len(selected) < target_count:
        made_progress = False
        for _fraction, method in sorted(remainders, key=lambda item: (-item[0], item[1])):
            remaining = [item for item in grouped[method] if item.relative_path not in selected_paths]
            if not remaining:
                continue
            shuffled = list(remaining)
            random.Random(f"{seed}:remainder:{method}:{len(selected)}").shuffle(shuffled)
            chosen = min(shuffled, key=lambda item: item.relative_path)
            selected.append(chosen)
            selected_paths.add(chosen.relative_path)
            made_progress = True
            if len(selected) == target_count:
                break
        if not made_progress:
            raise RuntimeError("Unable to complete FakeAVCeleb method-aware sampling.")

    return sorted(selected, key=lambda item: item.relative_path)


def sample_balanced_binary_records(records: list[FakeAVCelebRecord], seed: int) -> list[FakeAVCelebRecord]:
    real_records = sorted((record for record in records if record.label == 0), key=lambda item: item.relative_path)
    fake_records = [record for record in records if record.label == 1]
    sampled_fake = _sample_fake_records_by_method(fake_records, target_count=len(real_records), seed=seed)
    return sorted(real_records + sampled_fake, key=lambda item: item.relative_path)


def split_records(records: list[FakeAVCelebRecord], seed: int) -> dict[str, list[FakeAVCelebRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    split_map = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    return {
        split_name: sorted(split_rows, key=lambda item: item.relative_path)
        for split_name, split_rows in split_map.items()
    }


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
        "split_strategy": "random_video_level",
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
