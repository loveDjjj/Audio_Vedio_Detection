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
from src.data.mavos_dd_subset import build_english_subset_splits, write_split_csv, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build English-only MAVOS-DD subset splits for quick multi-generator experiments.")
    parser.add_argument("--metadata-root", type=Path, default=Path("dataset/MAVOS-DD-meta"), help="Directory containing MAVOS-DD metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("splits/mavos_dd_english_small"), help="Output directory for sampled split CSVs.")
    parser.add_argument("--train-ratio", type=float, default=0.2, help="Fraction of English train records to keep, stratified by generator.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of English open-set-model test records to keep, stratified by generator.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_mavos_dd_records(args.metadata_root)
    split_map = build_english_subset_splits(records, train_ratio=args.train_ratio, test_ratio=args.test_ratio, seed=args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_split_csv(output_dir / "train.csv", split_map["train"])
    write_split_csv(output_dir / "val.csv", split_map["val"])
    write_split_csv(output_dir / "test.csv", split_map["test"])

    summary = {
        "dataset": "MAVOS-DD",
        "language": "english",
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
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
