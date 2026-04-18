from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fakeavceleb_subset import (
    build_summary,
    load_fakeavceleb_records,
    select_real_fullfake_records,
    split_records,
    write_split_csv,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full FakeAVCeleb real/real vs fake/fake train/val/test CSV splits."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/OneDay/FakeAVCeleb"),
        help="FakeAVCeleb root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/fakeavceleb_real_fullfake"),
        help="Directory for generated split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic split generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    source_records = load_fakeavceleb_records(root)
    selected_records = select_real_fullfake_records(source_records)
    split_map = split_records(selected_records, seed=args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in split_map.items():
        write_split_csv(output_dir / f"{split_name}.csv", split_rows)

    summary = build_summary(
        split_map=split_map,
        source_records=source_records,
        selected_records=selected_records,
        seed=args.seed,
    )
    write_summary(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
