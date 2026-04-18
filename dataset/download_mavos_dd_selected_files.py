from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only the MAVOS-DD files referenced by sampled CSV splits.")
    parser.add_argument("--split-dir", type=Path, default=Path("splits/mavos_dd_english_small"), help="Directory containing train/val/test CSVs.")
    parser.add_argument("--output-root", type=Path, default=Path("/data/OneDay/MAVOS-DD"), help="Directory to download selected files into.")
    parser.add_argument("--repo-id", default="unibuc-cs/MAVOS-DD", help="Hugging Face dataset repo id.")
    return parser.parse_args()


def collect_relative_paths(split_dir: Path) -> list[str]:
    relative_paths: set[str] = set()
    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relative_paths.add(row["relative_path"])
    return sorted(relative_paths)


def main() -> int:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = build_logger(
        name="mavos-dd-download",
        level="INFO",
        log_file=output_root / "download.log",
        console=True,
    )

    relative_paths = collect_relative_paths(args.split_dir.resolve())
    summary = {
        "requested_files": len(relative_paths),
        "downloaded_files": 0,
        "existing_files": 0,
        "failed_files": [],
    }

    logger.info("Downloading %s selected MAVOS-DD files into %s", len(relative_paths), output_root)
    for relative_path in tqdm(relative_paths, desc="mavos-dd-download", leave=False):
        target_path = output_root / relative_path
        if target_path.exists():
            summary["existing_files"] += 1
            continue
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=relative_path,
                local_dir=str(output_root),
            )
            summary["downloaded_files"] += 1
        except Exception as exc:
            summary["failed_files"].append({"relative_path": relative_path, "reason": str(exc)})

    (output_root / "download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Download summary: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
