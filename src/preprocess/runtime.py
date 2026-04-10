from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty
from typing import Any

from tqdm import tqdm

from src.preprocess.manifest_builder import build_manifests
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


def build_worker_assignments(devices: list[int], workers_per_device: int) -> list[dict[str, int]]:
    if not devices:
        raise ValueError("runtime.devices must contain at least one CUDA device index.")
    if workers_per_device < 1:
        raise ValueError("runtime.workers_per_device must be >= 1.")

    nshard = len(devices) * workers_per_device
    assignments: list[dict[str, int]] = []
    rank = 0
    for _ in range(workers_per_device):
        for device_index in devices:
            assignments.append(
                {
                    "rank": rank,
                    "nshard": nshard,
                    "device_index": int(device_index),
                }
            )
            rank += 1
    return assignments


def build_worker_environment(device_index: int, cpu_threads_per_worker: int) -> dict[str, str]:
    thread_count = str(cpu_threads_per_worker)
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(device_index),
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
    }


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


def _validate_devices(devices: list[int]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict CNN preprocessing, but `torch.cuda.is_available()` is false.")
    device_count = torch.cuda.device_count()
    invalid = [device for device in devices if device < 0 or device >= device_count]
    if invalid:
        raise ValueError(
            f"runtime.devices contains invalid CUDA indices {invalid}; visible device count is {device_count}."
        )


def _emit_progress(progress_queue, worker_rank: int) -> None:
    if progress_queue is None:
        return
    progress_queue.put({"rank": worker_rank, "processed": 1})


def _run_preprocess_shard(
    config_path: str,
    rank: int,
    nshard: int,
    device_index: int,
    cpu_threads_per_worker: int,
    progress_queue=None,
) -> dict:
    worker_env = build_worker_environment(device_index=device_index, cpu_threads_per_worker=cpu_threads_per_worker)
    os.environ.update(worker_env)

    from src.preprocess.mouth_roi import process_manifest

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
        show_progress=False,
        progress_callback=lambda _file_id: _emit_progress(progress_queue, rank),
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
    summary["device_index"] = device_index
    return summary


def run_preprocess_from_config(config_path: Path = DEFAULT_PREPROCESS_CONFIG_PATH) -> dict:
    config = load_config(config_path)
    runtime_cfg, manifest_path = _resolve_runtime(config)
    artifact_root = ensure_dir(config["paths"]["artifact_root"])

    devices = [int(device) for device in runtime_cfg["devices"]]
    workers_per_device = int(runtime_cfg["workers_per_device"])
    cpu_threads_per_worker = int(runtime_cfg["cpu_threads_per_worker"])
    show_main_progress = bool(runtime_cfg.get("show_main_progress", True))
    _validate_devices(devices)
    assignments = build_worker_assignments(devices=devices, workers_per_device=workers_per_device)

    total_files = len(
        [
            line
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )

    if len(assignments) == 1:
        shard_summaries = [
            _run_preprocess_shard(
                config_path=str(config_path),
                rank=assignments[0]["rank"],
                nshard=assignments[0]["nshard"],
                device_index=assignments[0]["device_index"],
                cpu_threads_per_worker=cpu_threads_per_worker,
                progress_queue=None,
            )
        ]
    else:
        mp_context = mp.get_context(runtime_cfg.get("start_method", "spawn"))
        manager = mp_context.Manager()
        progress_queue = manager.Queue()
        progress_bar = (
            tqdm(total=total_files, desc=f"{config['preprocess']['stage']}:{config['preprocess']['manifest_name']}")
            if show_main_progress
            else None
        )

        try:
            with ProcessPoolExecutor(max_workers=len(assignments), mp_context=mp_context) as executor:
                futures = [
                    executor.submit(
                        _run_preprocess_shard,
                        str(config_path),
                        assignment["rank"],
                        assignment["nshard"],
                        assignment["device_index"],
                        cpu_threads_per_worker,
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
    combined_summary["devices"] = devices
    combined_summary["workers_per_device"] = workers_per_device
    combined_summary["num_procs"] = len(assignments)
    combined_summary["cpu_threads_per_worker"] = cpu_threads_per_worker
    combined_summary["start_method"] = runtime_cfg.get("start_method", "spawn")
    return combined_summary
