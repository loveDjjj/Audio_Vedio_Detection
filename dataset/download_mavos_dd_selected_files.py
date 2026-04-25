from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only the MAVOS-DD files referenced by sampled CSV splits.")
    parser.add_argument("--split-dir", type=Path, default=Path("splits/mavos_dd_real_fullfake"), help="Directory containing train/val/test CSVs.")
    parser.add_argument("--output-root", type=Path, default=Path("/data/OneDay/MAVOS-DD"), help="Directory to download selected files into.")
    parser.add_argument("--repo-id", default="unibuc-cs/MAVOS-DD", help="Hugging Face dataset repo id.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel file downloads. Use 1 for serial downloading.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def collect_relative_paths(split_dir: Path) -> list[str]:
    relative_paths: set[str] = set()
    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relative_paths.add(row["relative_path"])
    return sorted(relative_paths)


DownloadFn = Callable[..., str]


def _download_one(relative_path: str, *, output_root: Path, repo_id: str, download_fn: DownloadFn) -> dict[str, object]:
    target_path = output_root / relative_path
    if target_path.exists():
        return {"status": "existing", "relative_path": relative_path}
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        download_fn(
            repo_id=repo_id,
            repo_type="dataset",
            filename=relative_path,
            local_dir=str(output_root),
        )
        return {"status": "downloaded", "relative_path": relative_path}
    except Exception as exc:
        return {"status": "failed", "relative_path": relative_path, "reason": str(exc)}


def download_selected_files(
    relative_paths: list[str],
    *,
    output_root: Path,
    repo_id: str,
    download_fn: DownloadFn,
    workers: int,
    show_progress: bool = True,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    summary: dict[str, object] = {
        "requested_files": len(relative_paths),
        "downloaded_files": 0,
        "existing_files": 0,
        "failed_files": [],
    }

    def task(relative_path: str) -> dict[str, object]:
        return _download_one(relative_path, output_root=output_root, repo_id=repo_id, download_fn=download_fn)

    def consume_results(results: Iterable[dict[str, object]]) -> None:
        for result in tqdm(results, desc="mavos-dd-download", total=len(relative_paths), leave=False, disable=not show_progress):
            status = result["status"]
            if status == "existing":
                summary["existing_files"] = int(summary["existing_files"]) + 1
            elif status == "downloaded":
                summary["downloaded_files"] = int(summary["downloaded_files"]) + 1
            elif status == "failed":
                failed_files = summary["failed_files"]
                assert isinstance(failed_files, list)
                failed_files.append({"relative_path": result["relative_path"], "reason": result["reason"]})

    if workers == 1:
        consume_results(map(task, relative_paths))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(task, relative_path) for relative_path in relative_paths]
            consume_results(future.result() for future in as_completed(futures))

    return summary


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
    logger.info(
        "Downloading %s selected MAVOS-DD files into %s with workers=%s",
        len(relative_paths),
        output_root,
        args.workers,
    )
    summary = download_selected_files(
        relative_paths,
        output_root=output_root,
        repo_id=args.repo_id,
        download_fn=hf_hub_download,
        workers=args.workers,
    )

    (output_root / "download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Download summary: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
