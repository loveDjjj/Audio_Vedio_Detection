from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mavos_dd_metadata import load_mavos_dd_records
from src.data.mavos_dd_subset import (
    build_local_available_real_fullfake_official_splits,
    write_split_csv,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MAVOS-DD real/real vs fake/fake CSV splits from metadata, keeping only videos present locally."
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/data/OneDay/MAVOS-DD"),
        help="Directory containing MAVOS-DD metadata Arrow files.",
    )
    parser.add_argument(
        "--raw-video-root",
        type=Path,
        default=Path("/data/OneDay/MAVOS-DD"),
        help="Directory containing local MAVOS-DD mp4 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/mavos_dd_real_fullfake_local_available"),
        help="Output directory for generated local-available split CSVs.",
    )
    return parser.parse_args()


def _split_summary(records: list[dict]) -> dict:
    return {
        "videos": len(records),
        "labels": dict(Counter(record["label"] for record in records)),
        "generative_methods": dict(Counter(record["generative_method"] for record in records)),
    }


def main() -> int:
    args = parse_args()
    records = load_mavos_dd_records(args.metadata_root)
    split_map, availability = build_local_available_real_fullfake_official_splits(records, args.raw_video_root)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_split_csv(output_dir / "train.csv", split_map["train"])
    write_split_csv(output_dir / "val.csv", split_map["val"])
    write_split_csv(output_dir / "test.csv", split_map["test"])

    summary = {
        "dataset": "MAVOS-DD",
        "selection": {
            "real": {"label": "real", "audio_fake": False, "video_fake": False},
            "fake": {"label": "fake", "audio_fake": True, "video_fake": True},
        },
        "split_strategy": "official_split_local_available",
        "source_videos": len(records),
        "availability": availability,
        "splits": {split_name: _split_summary(split_records) for split_name, split_records in split_map.items()},
    }
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
