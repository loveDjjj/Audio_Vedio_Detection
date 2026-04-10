from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_features import compute_logfbank_features, resolve_audio_feature_path
from src.utils.logging_utils import build_logger
from src.utils.project import ensure_dir, load_config, resolve_path


def load_relative_paths(split_dir: Path) -> list[str]:
    import csv

    relative_paths: list[str] = []
    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            relative_paths.extend(row["relative_path"] for row in reader)
    return sorted(set(relative_paths))


def main() -> int:
    config = load_config(Path("configs/avhubert_classifier.yaml"))
    paths = config["paths"]
    logging_cfg = config["logging"]

    raw_video_root = resolve_path(paths["raw_video_root"])
    split_dir = resolve_path(paths["split_dir"])
    audio_feature_root = ensure_dir(paths["audio_feature_root"])
    ffmpeg = resolve_path(paths["ffmpeg_path"]) if paths.get("ffmpeg_path") else None
    ffmpeg_bin = str(ffmpeg) if ffmpeg is not None else "ffmpeg"

    logger = build_logger(
        name="audio-cache",
        level=logging_cfg["level"],
        log_file=audio_feature_root.parent / "audio_cache.log",
        console=True,
    )

    relative_paths = load_relative_paths(split_dir)
    summary = {
        "requested_files": len(relative_paths),
        "written_features": 0,
        "skipped_existing_features": 0,
        "failed_missing_video": 0,
        "failed_feature_extract": 0,
        "failed_files": [],
    }

    logger.info("Caching audio features to %s", audio_feature_root)
    for relative_path in tqdm(relative_paths, desc="audio-cache", leave=False):
        raw_video_path = raw_video_root / relative_path
        feature_path = resolve_audio_feature_path(audio_feature_root, relative_path)

        if not raw_video_path.exists():
            summary["failed_missing_video"] += 1
            summary["failed_files"].append({"relative_path": relative_path, "reason": "missing_video"})
            continue
        if feature_path.exists():
            summary["skipped_existing_features"] += 1
            continue

        try:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            features = compute_logfbank_features(raw_video_path=raw_video_path, ffmpeg=ffmpeg_bin)
            import numpy as np

            np.save(feature_path, features)
            summary["written_features"] += 1
        except Exception as exc:
            summary["failed_feature_extract"] += 1
            summary["failed_files"].append({"relative_path": relative_path, "reason": f"feature_extract_failed:{exc}"})

    logger.info("Audio cache summary: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
