from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoRecord:
    relative_path: str
    label: int
    label_name: str
    clip_key: str
    person_id: str
    source_video_id: str
    clip_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val/test lists from AV-Deepfake1M val split using only "
            "real.mp4 and fake_video_fake_audio.mp4 with random video-level splitting."
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


def build_video_records(root: Path) -> tuple[list[VideoRecord], dict[str, int]]:
    metadata_path = root / "val_metadata.json"
    val_dir = root / "val"
    data = load_metadata(metadata_path)

    records: list[VideoRecord] = []
    stats = {
        "total_metadata_rows": len(data),
        "real_rows": 0,
        "fullfake_rows": 0,
        "missing_video_files": 0,
    }

    for item in data:
        rel_path = item["file"]
        filename = rel_path.split("/")[-1]
        if filename == "real.mp4":
            label = 0
            label_name = "real"
            stats["real_rows"] += 1
        elif filename == "fake_video_fake_audio.mp4":
            label = 1
            label_name = "fake_video_fake_audio"
            stats["fullfake_rows"] += 1
        else:
            continue

        full_path = val_dir / rel_path
        if not full_path.exists():
            stats["missing_video_files"] += 1
            continue

        person_id, source_video_id, clip_id = rel_path.split("/")[:3]
        records.append(
            VideoRecord(
                relative_path=rel_path,
                label=label,
                label_name=label_name,
                clip_key="/".join(rel_path.split("/")[:3]),
                person_id=person_id,
                source_video_id=source_video_id,
                clip_id=clip_id,
            )
        )

    stats["selected_videos"] = len(records)
    return records, stats


def split_records(records: list[VideoRecord], seed: int) -> dict[str, list[VideoRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    split_records = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    for split_name in split_records:
        split_records[split_name].sort(key=lambda record: record.relative_path)
    return split_records


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
    stats: dict[str, int],
    seed: int,
) -> dict:
    summary = {
        "dataset": "AV-Deepfake1M",
        "source_split": "val",
        "selection": ["real.mp4", "fake_video_fake_audio.mp4"],
        "split_strategy": "random_video_level",
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
