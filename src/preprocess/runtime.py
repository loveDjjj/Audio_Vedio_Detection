from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.preprocess.manifest_builder import build_manifests
from src.preprocess.mouth_roi import process_manifest
from src.utils.project import ensure_dir, load_config, resolve_path


DEFAULT_PREPROCESS_CONFIG_PATH = Path("configs/avhubert_preprocess.yaml")

SUMMARY_KEYS = [
    "requested_files",
    "mouth_roi_written",
    "landmarks_written",
    "skipped_existing_mouth_roi",
    "skipped_existing_landmarks",
    "failed_missing_video",
    "failed_missing_landmarks",
    "failed_no_landmarks",
    "failed_crop",
]


def build_worker_shards(num_procs: int) -> list[dict[str, int]]:
    if num_procs < 1:
        raise ValueError("runtime.num_procs must be >= 1")
    return [{"rank": rank, "nshard": num_procs} for rank in range(num_procs)]


def aggregate_shard_summaries(shard_summaries: list[dict]) -> dict:
    if not shard_summaries:
        raise ValueError("No shard summaries to aggregate.")

    first = shard_summaries[0]
    summary = {
        "stage": first["stage"],
        "manifest": first["manifest"],
        "num_shards": len(shard_summaries),
        "failed_files": [],
    }
    for key in SUMMARY_KEYS:
        summary[key] = sum(int(item.get(key, 0)) for item in shard_summaries)
    for item in shard_summaries:
        summary["failed_files"].extend(item.get("failed_files", []))
    return summary


def _resolve_runtime(config: dict) -> tuple[dict, Path]:
    paths = config["paths"]
    preprocess_cfg = config["preprocess"]
    runtime_cfg = config["runtime"]

    split_dir = resolve_path(paths["split_dir"])
    manifest_dir = ensure_dir(paths["manifest_dir"])
    build_manifests(
        split_dir=split_dir,
        output_dir=manifest_dir,
        split_names=preprocess_cfg["split_names"],
    )

    manifest_path = manifest_dir / f"{preprocess_cfg['manifest_name']}.list"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return runtime_cfg, manifest_path


def _run_preprocess_shard(config_path: str, rank: int, nshard: int, show_progress: bool) -> dict:
    config = load_config(Path(config_path))
    paths = config["paths"]
    preprocess_cfg = config["preprocess"]

    manifest_path = resolve_path(paths["manifest_dir"]) / f"{preprocess_cfg['manifest_name']}.list"
    artifact_root = ensure_dir(paths["artifact_root"])

    summary = process_manifest(
        raw_video_root=resolve_path(paths["raw_video_root"]),
        manifest_path=manifest_path,
        landmark_root=ensure_dir(paths["landmark_dir"]),
        mouth_roi_root=ensure_dir(paths["mouth_roi_root"]),
        face_predictor_path=resolve_path(paths["face_predictor_path"]),
        mean_face_path=resolve_path(paths["mean_face_path"]),
        cnn_detector_path=resolve_path(paths["cnn_detector_path"]),
        rank=rank,
        nshard=nshard,
        crop_width=preprocess_cfg["crop_width"],
        crop_height=preprocess_cfg["crop_height"],
        start_idx=preprocess_cfg["start_idx"],
        stop_idx=preprocess_cfg["stop_idx"],
        window_margin=preprocess_cfg["window_margin"],
        fps=preprocess_cfg["fps"],
        stage=preprocess_cfg["stage"],
        save_landmarks=preprocess_cfg["save_landmarks"],
        strict=preprocess_cfg["strict"],
        show_progress=show_progress,
    )

    suffix = f"{preprocess_cfg['stage']}_{preprocess_cfg['manifest_name']}_rank{rank:03d}_of_{nshard:03d}"
    summary_path = artifact_root / f"preprocess_{suffix}.json"
    failed_path = artifact_root / f"preprocess_{suffix}.failed.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    failed_path.write_text(
        "\n".join(item["file_id"] for item in summary["failed_files"]) + ("\n" if summary["failed_files"] else ""),
        encoding="utf-8",
    )
    return summary


def run_preprocess_from_config(config_path: Path = DEFAULT_PREPROCESS_CONFIG_PATH) -> dict:
    config = load_config(config_path)
    runtime_cfg, manifest_path = _resolve_runtime(config)
    artifact_root = ensure_dir(config["paths"]["artifact_root"])

    num_procs = int(runtime_cfg["num_procs"])
    show_worker_progress = bool(runtime_cfg.get("show_worker_progress", num_procs == 1))
    shards = build_worker_shards(num_procs)

    if num_procs == 1:
        shard_summaries = [
            _run_preprocess_shard(
                config_path=str(config_path),
                rank=0,
                nshard=1,
                show_progress=show_worker_progress,
            )
        ]
    else:
        mp_context = mp.get_context(runtime_cfg.get("start_method", "spawn"))
        with ProcessPoolExecutor(max_workers=num_procs, mp_context=mp_context) as executor:
            futures = [
                executor.submit(
                    _run_preprocess_shard,
                    str(config_path),
                    shard["rank"],
                    shard["nshard"],
                    show_worker_progress,
                )
                for shard in shards
            ]
            shard_summaries = [future.result() for future in futures]

    combined_summary = aggregate_shard_summaries(shard_summaries)
    suffix = f"{config['preprocess']['stage']}_{config['preprocess']['manifest_name']}"
    summary_path = artifact_root / f"preprocess_{suffix}.json"
    failed_path = artifact_root / f"preprocess_{suffix}.failed.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, indent=2, ensure_ascii=False)
    failed_path.write_text(
        "\n".join(item["file_id"] for item in combined_summary["failed_files"])
        + ("\n" if combined_summary["failed_files"] else ""),
        encoding="utf-8",
    )

    combined_summary["config_path"] = str(resolve_path(config_path))
    combined_summary["manifest_path"] = str(manifest_path)
    combined_summary["num_procs"] = num_procs
    combined_summary["start_method"] = runtime_cfg.get("start_method", "spawn")
    return combined_summary
