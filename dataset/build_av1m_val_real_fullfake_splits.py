from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoRecord:
    relative_path: str
    label: int
    label_name: str
    source_split: str
    clip_key: str
    person_id: str
    source_video_id: str
    clip_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build AV-Deepfake1M real/real vs fake/fake CSV splits from the official "
            "train_metadata.json and val_metadata.json protocol."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/OneDay/AV-Deepfake1M"),
        help="AV-Deepfake1M root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/av1m_val_real_fullfake"),
        help="Directory for generated split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic split generation.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_label(item: dict) -> tuple[int, str] | None:
    rel_path = str(item["file"])
    filename = Path(rel_path).name
    modify_type = str(item.get("modify_type", ""))
    if filename == "real.mp4" or modify_type == "real":
        return 0, "real"
    if filename == "fake_video_fake_audio.mp4" or modify_type == "both_modified":
        return 1, "fake_video_fake_audio"
    return None


def _build_records_for_split(root: Path, source_split: str) -> tuple[list[VideoRecord], dict[str, int]]:
    metadata_path = root / f"{source_split}_metadata.json"
    data = load_metadata(metadata_path)

    records: list[VideoRecord] = []
    stats = {
        "total_metadata_rows": len(data),
        "real_rows": 0,
        "fullfake_rows": 0,
        "missing_video_files": 0,
    }

    for item in data:
        label_info = _select_label(item)
        if label_info is None:
            continue

        rel_path = str(item["file"]).strip()
        relative_path = f"{source_split}/{rel_path}"
        label, label_name = label_info
        if label == 0:
            stats["real_rows"] += 1
        else:
            stats["fullfake_rows"] += 1

        full_path = root / relative_path
        if not full_path.exists():
            stats["missing_video_files"] += 1
            continue

        person_id, source_video_id, clip_id = rel_path.split("/")[:3]
        records.append(
            VideoRecord(
                relative_path=relative_path,
                label=label,
                label_name=label_name,
                source_split=source_split,
                clip_key=f"{source_split}/" + "/".join(rel_path.split("/")[:3]),
                person_id=person_id,
                source_video_id=source_video_id,
                clip_id=clip_id,
            )
        )

    stats["selected_videos"] = len(records)
    return sorted(records, key=lambda record: record.relative_path), stats


def build_video_records(root: Path) -> tuple[list[VideoRecord], dict[str, dict[str, int]]]:
    records: list[VideoRecord] = []
    source_stats: dict[str, dict[str, int]] = {}
    for source_split in ("train", "val"):
        split_records, split_stats = _build_records_for_split(root, source_split)
        records.extend(split_records)
        source_stats[source_split] = split_stats
    return records, source_stats


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


def split_records(records: list[VideoRecord], seed: int) -> dict[str, list[VideoRecord]]:
    train_records = sorted((record for record in records if record.source_split == "train"), key=lambda item: item.relative_path)
    val_source_records = [record for record in records if record.source_split == "val"]

    grouped_val: dict[str, list[VideoRecord]] = defaultdict(list)
    for record in val_source_records:
        grouped_val[record.clip_key].append(record)

    clip_keys = sorted(grouped_val)
    random.Random(seed).shuffle(clip_keys)
    val_group_count, test_group_count = _allocate_counts(len(clip_keys), (0.5, 0.5))
    val_keys = set(clip_keys[:val_group_count])
    test_keys = set(clip_keys[val_group_count : val_group_count + test_group_count])

    split_map = {
        "train": train_records,
        "val": sorted(
            (record for clip_key in val_keys for record in grouped_val[clip_key]),
            key=lambda item: item.relative_path,
        ),
        "test": sorted(
            (record for clip_key in test_keys for record in grouped_val[clip_key]),
            key=lambda item: item.relative_path,
        ),
    }
    return split_map


def write_rows(path: Path, records: list[VideoRecord]) -> None:
    fieldnames = [
        "relative_path",
        "label",
        "label_name",
        "clip_key",
        "person_id",
        "source_video_id",
        "clip_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "relative_path": record.relative_path,
                    "label": record.label,
                    "label_name": record.label_name,
                    "clip_key": record.clip_key,
                    "person_id": record.person_id,
                    "source_video_id": record.source_video_id,
                    "clip_id": record.clip_id,
                }
            )


def build_summary(
    split_records: dict[str, list[VideoRecord]],
    stats: dict[str, dict[str, int]],
    seed: int,
) -> dict:
    summary = {
        "dataset": "AV-Deepfake1M",
        "source_splits": ["train", "val"],
        "selection": ["real.mp4", "fake_video_fake_audio.mp4"],
        "split_strategy": "official_train_plus_val_clip_holdout",
        "seed": seed,
        "source_stats": stats,
        "splits": {},
    }
    for split_name, records in split_records.items():
        summary["splits"][split_name] = {
            "videos": len(records),
            "real_videos": sum(record.label == 0 for record in records),
            "fake_videos": sum(record.label == 1 for record in records),
        }
    return summary


def main() -> int:
    args = parse_args()
    records, stats = build_video_records(args.root.resolve())
    split_records_map = split_records(records, args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in split_records_map.items():
        write_rows(output_dir / f"{split_name}.csv", split_rows)

    summary = build_summary(split_records_map, stats, args.seed)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
