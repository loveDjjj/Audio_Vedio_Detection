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
from src.data.mavos_dd_subset import build_real_fullfake_official_splits, write_split_csv, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MAVOS-DD official real/real vs fake/fake train/val/test CSV splits."
    )
    parser.add_argument("--metadata-root", type=Path, default=Path("/data/OneDay/MAVOS-DD"), help="Directory containing MAVOS-DD metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("splits/mavos_dd_english_small"), help="Output directory for generated split CSVs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_mavos_dd_records(args.metadata_root)
    split_map = build_real_fullfake_official_splits(records)

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
        "split_strategy": "official_split",
        "source_videos": len(records),
        "splits": {},
    }
    for split_name, records_ in split_map.items():
        summary["splits"][split_name] = {
            "videos": len(records_),
            "labels": dict(Counter(record["label"] for record in records_)),
            "generative_methods": dict(Counter(record["generative_method"] for record in records_)),
        }
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
