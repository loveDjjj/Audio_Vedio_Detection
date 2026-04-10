from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.manifest_builder import build_manifests
from src.preprocess.mouth_roi import process_manifest
from src.utils.project import ensure_dir, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict mouth ROI preprocessing for the selected AV1M val subset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/avhubert_classifier.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "detect", "align"],
        default="all",
        help="Preprocessing stage to execute.",
    )
    parser.add_argument(
        "--manifest-name",
        default="all",
        help="Manifest name to process. One of train/val/test/all.",
    )
    parser.add_argument("--rank", type=int, default=None, help="Shard rank override.")
    parser.add_argument("--nshard", type=int, default=None, help="Total shard count override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    paths = config["paths"]
    preprocess_cfg = config["preprocess"]

    split_dir = resolve_path(paths["split_dir"])
    manifest_dir = ensure_dir(paths["manifest_dir"])
    build_manifests(
        split_dir=split_dir,
        output_dir=manifest_dir,
        split_names=preprocess_cfg["manifest_names"],
    )

    raw_video_root = resolve_path(paths["raw_video_root"])
    landmark_root = ensure_dir(paths["landmark_dir"])
    mouth_roi_root = ensure_dir(paths["mouth_roi_root"])
    artifact_root = ensure_dir(paths["artifact_root"])
    manifest_path = manifest_dir / f"{args.manifest_name}.list"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    face_predictor_path = resolve_path(paths["face_predictor_path"])
    mean_face_path = resolve_path(paths["mean_face_path"])
    cnn_detector_path = resolve_path(paths["cnn_detector_path"])
    if not face_predictor_path.exists():
        raise FileNotFoundError(f"Missing face predictor: {face_predictor_path}")
    if not mean_face_path.exists():
        raise FileNotFoundError(f"Missing mean face file: {mean_face_path}")
    if not cnn_detector_path.exists():
        raise FileNotFoundError(
            f"Missing CNN face detector: {cnn_detector_path}. "
            "Strict preprocessing now requires the GPU CNN detector and does not fall back to HOG."
        )
    cnn_detector = cnn_detector_path

    rank = preprocess_cfg["rank"] if args.rank is None else args.rank
    nshard = preprocess_cfg["nshard"] if args.nshard is None else args.nshard

    summary = process_manifest(
        raw_video_root=raw_video_root,
        manifest_path=manifest_path,
        landmark_root=landmark_root,
        mouth_roi_root=mouth_roi_root,
        face_predictor_path=face_predictor_path,
        mean_face_path=mean_face_path,
        cnn_detector_path=cnn_detector,
        rank=rank,
        nshard=nshard,
        crop_width=preprocess_cfg["crop_width"],
        crop_height=preprocess_cfg["crop_height"],
        start_idx=preprocess_cfg["start_idx"],
        stop_idx=preprocess_cfg["stop_idx"],
        window_margin=preprocess_cfg["window_margin"],
        fps=preprocess_cfg["fps"],
        stage=args.stage,
        save_landmarks=preprocess_cfg["save_landmarks"],
        strict=preprocess_cfg["strict"],
    )

    suffix = f"{args.stage}_{args.manifest_name}_rank{rank:03d}_of_{nshard:03d}"
    summary_path = artifact_root / f"preprocess_{suffix}.json"
    failed_path = artifact_root / f"preprocess_{suffix}.failed.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    failed_path.write_text(
        "\n".join(item["file_id"] for item in summary["failed_files"]) + ("\n" if summary["failed_files"] else ""),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
