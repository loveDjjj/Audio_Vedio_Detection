from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mavos_dd_metadata import (
    load_mavos_dd_records,
    summarize_mavos_dd_records,
    write_mavos_dd_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MAVOS-DD metadata and write a JSON summary.")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("dataset/MAVOS-DD-meta"),
        help="Directory containing MAVOS-DD metadata files including data-*.arrow.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <metadata-root>/mavos_dd_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.output or (args.metadata_root / "mavos_dd_summary.json")

    records = load_mavos_dd_records(args.metadata_root)
    summary = summarize_mavos_dd_records(records)
    write_mavos_dd_summary(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
