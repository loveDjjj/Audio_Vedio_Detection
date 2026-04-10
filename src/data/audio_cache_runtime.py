from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty

import numpy as np
from tqdm import tqdm

from src.data.audio_features import (
    compute_logfbank_features,
    resolve_audio_feature_path,
    stack_audio_features,
)
from src.utils.logging_utils import build_logger
from src.utils.project import ensure_dir, load_config, resolve_path


SUMMARY_KEYS = [
    "requested_files",
    "written_features",
    "skipped_existing_features",
    "failed_missing_video",
    "failed_feature_extract",
]


def build_audio_cache_assignments(num_procs: int) -> list[dict[str, int]]:
    if num_procs < 1:
        raise ValueError("audio_cache.num_procs must be >= 1")
    return [{"rank": rank, "nshard": num_procs} for rank in range(num_procs)]


def split_relative_paths_for_rank(relative_paths: list[str], rank: int, nshard: int) -> list[str]:
    if rank < 0 or rank >= nshard:
        raise ValueError(f"rank must be in [0, {nshard - 1}], got {rank}")
    return relative_paths[rank::nshard]


def aggregate_audio_cache_summaries(shard_summaries: list[dict]) -> dict:
    if not shard_summaries:
        raise ValueError("No shard summaries to aggregate.")

    summary = {
        "num_shards": len(shard_summaries),
        "failed_files": [],
    }
    for key in SUMMARY_KEYS:
        summary[key] = sum(int(item.get(key, 0)) for item in shard_summaries)
    for item in shard_summaries:
        summary["failed_files"].extend(item.get("failed_files", []))
    return summary


def _load_relative_paths(split_dir: Path) -> list[str]:
    import csv

    relative_paths: list[str] = []
    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            relative_paths.extend(row["relative_path"] for row in reader)
    return sorted(set(relative_paths))


def _emit_progress(progress_queue) -> None:
    if progress_queue is None:
        return
    progress_queue.put({"processed": 1})


def _run_audio_cache_shard(
    config_path: str,
    rank: int,
    nshard: int,
    progress_queue=None,
) -> dict:
    config = load_config(Path(config_path))
    paths = config["paths"]
    logging_cfg = config["logging"]
    cache_cfg = config["audio_cache"]

    raw_video_root = resolve_path(paths["raw_video_root"])
    split_dir = resolve_path(paths["split_dir"])
    audio_feature_root = ensure_dir(paths["audio_feature_root"])
    ffmpeg = resolve_path(paths["ffmpeg_path"]) if paths.get("ffmpeg_path") else None
    ffmpeg_bin = str(ffmpeg) if ffmpeg is not None else "ffmpeg"
    relative_paths = split_relative_paths_for_rank(_load_relative_paths(split_dir), rank=rank, nshard=nshard)

    cpu_threads_per_worker = int(cache_cfg["cpu_threads_per_worker"])
    thread_count = str(cpu_threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count

    logger = build_logger(
        name=f"audio-cache.rank{rank}",
        level=logging_cfg["level"],
        log_file=audio_feature_root.parent / logging_cfg["audio_cache_rank_log_filename_template"].format(rank=rank),
        console=False,
    )
    logger.info("Starting audio cache shard rank=%s/%s with cpu_threads_per_worker=%s", rank, nshard, cpu_threads_per_worker)

    summary = {
        "requested_files": len(relative_paths),
        "written_features": 0,
        "skipped_existing_features": 0,
        "failed_missing_video": 0,
        "failed_feature_extract": 0,
        "failed_files": [],
    }

    for relative_path in relative_paths:
        raw_video_path = raw_video_root / relative_path
        feature_path = resolve_audio_feature_path(audio_feature_root, relative_path)

        if not raw_video_path.exists():
            summary["failed_missing_video"] += 1
            summary["failed_files"].append({"relative_path": relative_path, "reason": "missing_video"})
            _emit_progress(progress_queue)
            continue
        if feature_path.exists():
            summary["skipped_existing_features"] += 1
            _emit_progress(progress_queue)
            continue

        try:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            features = compute_logfbank_features(raw_video_path=raw_video_path, ffmpeg=ffmpeg_bin)
            stacked = stack_audio_features(features, int(cache_cfg["stack_order_audio"]))
            np.save(feature_path, stacked.astype(np.float32))
            summary["written_features"] += 1
        except Exception as exc:
            summary["failed_feature_extract"] += 1
            summary["failed_files"].append({"relative_path": relative_path, "reason": f"feature_extract_failed:{exc}"})
        finally:
            _emit_progress(progress_queue)

    logger.info("Completed audio cache shard summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def run_audio_cache_from_config(config_path: Path = Path("configs/avhubert_classifier.yaml")) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    logging_cfg = config["logging"]
    cache_cfg = config["audio_cache"]

    split_dir = resolve_path(paths["split_dir"])
    audio_feature_root = ensure_dir(paths["audio_feature_root"])
    relative_paths = _load_relative_paths(split_dir)
    num_procs = int(cache_cfg["num_procs"])
    show_progress = bool(cache_cfg.get("show_progress", True))

    logger = build_logger(
        name="audio-cache.main",
        level=logging_cfg["level"],
        log_file=audio_feature_root.parent / logging_cfg["audio_cache_log_filename"],
        console=True,
    )
    logger.info(
        "Audio cache config=%s num_procs=%s cpu_threads_per_worker=%s stack_order_audio=%s",
        resolve_path(config_path),
        num_procs,
        cache_cfg["cpu_threads_per_worker"],
        cache_cfg["stack_order_audio"],
    )

    assignments = build_audio_cache_assignments(num_procs)
    if num_procs == 1:
        shard_summaries = [_run_audio_cache_shard(str(config_path), rank=0, nshard=1, progress_queue=None)]
    else:
        mp_context = mp.get_context(cache_cfg.get("start_method", "spawn"))
        manager = mp_context.Manager()
        progress_queue = manager.Queue()
        progress_bar = tqdm(total=len(relative_paths), desc="audio-cache", leave=False) if show_progress else None
        try:
            with ProcessPoolExecutor(max_workers=num_procs, mp_context=mp_context) as executor:
                futures = [
                    executor.submit(
                        _run_audio_cache_shard,
                        str(config_path),
                        assignment["rank"],
                        assignment["nshard"],
                        progress_queue,
                    )
                    for assignment in assignments
                ]
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    while True:
                        try:
                            event = progress_queue.get_nowait()
                        except Empty:
                            break
                        if progress_bar is not None:
                            progress_bar.update(int(event.get("processed", 0)))
                    if done:
                        continue
                shard_summaries = [future.result() for future in futures]
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except Empty:
                        break
                    if progress_bar is not None:
                        progress_bar.update(int(event.get("processed", 0)))
        finally:
            if progress_bar is not None:
                progress_bar.close()
            manager.shutdown()

    combined_summary = aggregate_audio_cache_summaries(shard_summaries)
    combined_summary["config_path"] = str(resolve_path(config_path))
    combined_summary["num_procs"] = num_procs
    combined_summary["cpu_threads_per_worker"] = int(cache_cfg["cpu_threads_per_worker"])
    combined_summary["stack_order_audio"] = int(cache_cfg["stack_order_audio"])
    logger.info("Audio cache summary: %s", json.dumps(combined_summary, ensure_ascii=False))
    return combined_summary
