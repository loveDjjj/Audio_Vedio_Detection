from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from shutil import copy2

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.av1m_mouth_roi_dataset import AV1MMouthRoiDataset
from src.data.collate import collate_audio_video_batch
from src.models.avhubert_backbone import load_avhubert_checkpoint_metadata
from src.models.binary_detector import AVHubertBinaryDetector
from src.train.engine import run_epoch
from src.train.runtime import (
    build_distributed_config,
    distributed_enabled,
    is_main_process,
    resolve_training_devices,
    unwrap_model,
)
from src.utils.logging_utils import build_logger
from src.utils.project import ensure_dir, load_config, resolve_path, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen AV-HuBERT binary classifier on the mouth-ROI AV1M val subset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/avhubert_classifier.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def build_dataset(split_name: str, config: dict, training: bool, backbone_metadata: dict):
    paths = config["paths"]
    data_cfg = config["data"]
    return AV1MMouthRoiDataset(
        csv_path=resolve_path(paths["split_dir"]) / f"{split_name}.csv",
        raw_video_root=resolve_path(paths["raw_video_root"]),
        mouth_roi_root=resolve_path(paths["mouth_roi_root"]),
        avhubert_repo=resolve_path(paths["avhubert_repo"]),
        training=training,
        image_crop_size=data_cfg["image_crop_size"],
        image_mean=data_cfg["image_mean"],
        image_std=data_cfg["image_std"],
        horizontal_flip_prob=data_cfg["horizontal_flip_prob"],
        stack_order_audio=backbone_metadata["stack_order_audio"],
        normalize_audio=backbone_metadata["audio_normalize"],
    )


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_frames: int,
    pad_to_batch_max: bool,
    drop_last: bool,
    sampler=None,
):
    collate_fn = partial(
        collate_audio_video_batch,
        max_frames=max_frames,
        pad_to_batch_max=pad_to_batch_max,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def resolve_device(device_name: str) -> torch.device:
    if not device_name.startswith("cuda"):
        raise ValueError("This training entrypoint is GPU-only. Set the training device to a CUDA target such as `cuda:0`.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entrypoint, but `torch.cuda.is_available()` is false.")
    if device_name == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_name)


def validate_cuda_devices(devices: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but `torch.cuda.is_available()` is false.")
    device_count = torch.cuda.device_count()
    invalid = [device for device in devices if device < 0 or device >= device_count]
    if invalid:
        raise ValueError(f"train.devices contains invalid CUDA indices {invalid}; visible CUDA devices: {device_count}.")


def configure_cuda_runtime(device: torch.device) -> dict[str, str | int]:
    torch.cuda.set_device(device)
    # Use a conservative cudnn path because sequence lengths vary a lot batch-to-batch.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_index": int(device.index or 0),
    }


def setup_process_group(distributed_cfg) -> None:
    if distributed_cfg.world_size == 1:
        return
    dist.init_process_group(
        backend=distributed_cfg.backend,
        init_method=f"tcp://{distributed_cfg.master_addr}:{distributed_cfg.master_port}",
        rank=distributed_cfg.rank,
        world_size=distributed_cfg.world_size,
    )


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_head_checkpoint(path: Path, epoch: int, model, metrics: dict, config: dict) -> None:
    unwrapped_model = unwrap_model(model)
    state = {
        "epoch": epoch,
        "head_state_dict": unwrapped_model.head.state_dict(),
        "metrics": metrics,
        "config": config,
        "backbone_load_info": unwrapped_model.backbone.load_info,
    }
    torch.save(state, path)


def load_best_head_to_current_model(best_path: Path, model) -> None:
    unwrapped_model = unwrap_model(model)
    if dist.is_available() and dist.is_initialized():
        object_list = [None]
        if is_main_process(dist.get_rank()) and best_path.exists():
            best_state = torch.load(best_path, map_location="cpu")
            object_list[0] = best_state["head_state_dict"]
        dist.broadcast_object_list(object_list, src=0)
        if object_list[0] is not None:
            unwrapped_model.head.load_state_dict(object_list[0])
        return

    if best_path.exists():
        best_state = torch.load(best_path, map_location="cpu")
        unwrapped_model.head.load_state_dict(best_state["head_state_dict"])


def run_worker(local_rank: int, config_path: str, run_dir: str) -> None:
    config = load_config(Path(config_path))
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    paths = config["paths"]
    logging_cfg = config["logging"]

    distributed_cfg = build_distributed_config(train_cfg=train_cfg, local_rank=local_rank)
    device = resolve_device(f"cuda:{distributed_cfg.device_index}")
    setup_process_group(distributed_cfg)

    try:
        run_dir_path = Path(run_dir)
        logger = build_logger(
            name=f"train.rank{distributed_cfg.rank}",
            level=logging_cfg["level"],
            log_file=run_dir_path
            / (
                logging_cfg["train_log_filename"]
                if is_main_process(distributed_cfg.rank)
                else logging_cfg["rank_log_filename_template"].format(rank=distributed_cfg.rank)
            ),
            console=is_main_process(distributed_cfg.rank) or bool(logging_cfg.get("show_rank_logs", False)),
        )
        gpu_info = configure_cuda_runtime(device)
        if is_main_process(distributed_cfg.rank):
            if distributed_enabled(distributed_cfg.devices):
                logger.info(
                    f"Launching DDP on devices {distributed_cfg.devices} "
                    f"(world_size={distributed_cfg.world_size}, backend={distributed_cfg.backend})"
                )
            logger.info("Using GPU %s: %s (%s)", gpu_info["gpu_index"], gpu_info["gpu_name"], gpu_info["device"])

        seed_everything(train_cfg["seed"] + distributed_cfg.rank)

        checkpoint_path = resolve_path(paths["checkpoint_path"])
        backbone_metadata = load_avhubert_checkpoint_metadata(checkpoint_path)
        if not model_cfg["freeze_backbone"]:
            raise ValueError("SSR-DFD AV-HuBERT baseline requires `model.freeze_backbone: true`.")

        train_dataset = build_dataset("train", config, training=True, backbone_metadata=backbone_metadata)
        val_dataset = build_dataset("val", config, training=False, backbone_metadata=backbone_metadata)
        test_dataset = build_dataset("test", config, training=False, backbone_metadata=backbone_metadata)

        train_sampler = None
        val_sampler = None
        test_sampler = None
        if distributed_enabled(distributed_cfg.devices):
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=distributed_cfg.world_size,
                rank=distributed_cfg.rank,
                shuffle=data_cfg["shuffle_train"],
                drop_last=data_cfg["drop_last_train"],
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=distributed_cfg.world_size,
                rank=distributed_cfg.rank,
                shuffle=False,
                drop_last=False,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=distributed_cfg.world_size,
                rank=distributed_cfg.rank,
                shuffle=False,
                drop_last=False,
            )

        train_loader = make_loader(
            dataset=train_dataset,
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
            shuffle=data_cfg["shuffle_train"],
            max_frames=data_cfg["max_frames"],
            pad_to_batch_max=data_cfg["pad_to_batch_max"],
            drop_last=data_cfg["drop_last_train"],
            sampler=train_sampler,
        )
        eval_loader_kwargs = {
            "batch_size": train_cfg["batch_size"],
            "num_workers": train_cfg["num_workers"],
            "shuffle": False,
            "max_frames": data_cfg["max_frames"],
            "pad_to_batch_max": data_cfg["pad_to_batch_max"],
            "drop_last": False,
        }
        val_loader = make_loader(val_dataset, sampler=val_sampler, **eval_loader_kwargs)
        test_loader = make_loader(test_dataset, sampler=test_sampler, **eval_loader_kwargs)

        model = AVHubertBinaryDetector(
            checkpoint_path=checkpoint_path,
            avhubert_repo=resolve_path(paths["avhubert_repo"]),
            freeze_backbone=model_cfg["freeze_backbone"],
            feat_dim=backbone_metadata["encoder_embed_dim"],
        ).to(device)
        if distributed_enabled(distributed_cfg.devices):
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        criterion = BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"])

        best_path = run_dir_path / "best_head.pt"
        last_path = run_dir_path / "last_head.pt"
        history: list[dict] = []
        best_epoch = -1
        best_val_f1 = float("-inf")

        dataset_summary = {
            "train_videos": len(train_dataset),
            "val_videos": len(val_dataset),
            "test_videos": len(test_dataset),
            "train_missing_files": train_dataset.missing_files,
            "val_missing_files": val_dataset.missing_files,
            "test_missing_files": test_dataset.missing_files,
            "train_missing_mouth_roi": train_dataset.missing_mouth_roi_files,
            "val_missing_mouth_roi": val_dataset.missing_mouth_roi_files,
            "test_missing_mouth_roi": test_dataset.missing_mouth_roi_files,
            "train_missing_raw_video": train_dataset.missing_raw_video_files,
            "val_missing_raw_video": val_dataset.missing_raw_video_files,
            "test_missing_raw_video": test_dataset.missing_raw_video_files,
            "audio_stack_order": float(backbone_metadata["stack_order_audio"]),
            "audio_normalize": float(backbone_metadata["audio_normalize"]),
        }
        logger.info("Dataset summary: %s", dataset_summary)

        for epoch in range(1, train_cfg["epochs"] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main_process(distributed_cfg.rank):
                logger.info("Epoch %s/%s", epoch, train_cfg["epochs"])

            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=train_cfg["grad_clip_norm"],
                amp=train_cfg["amp"],
                log_interval=train_cfg["log_interval"],
                phase="train",
                epoch=epoch,
                logger=logger,
                show_progress=bool(logging_cfg.get("show_progress", True)),
            )

            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    scaler=None,
                    grad_clip_norm=0.0,
                    amp=train_cfg["amp"],
                    log_interval=0,
                    phase="val",
                    epoch=epoch,
                    logger=logger,
                    show_progress=bool(logging_cfg.get("show_progress", True)),
                )

            if is_main_process(distributed_cfg.rank):
                record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                history.append(record)
                logger.info("Epoch metrics: %s", json.dumps(record, ensure_ascii=False))

                save_head_checkpoint(last_path, epoch=epoch, model=model, metrics=val_metrics, config=config)
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_epoch = epoch
                    save_head_checkpoint(best_path, epoch=epoch, model=model, metrics=val_metrics, config=config)

        load_best_head_to_current_model(best_path, model)

        with torch.no_grad():
            test_metrics = run_epoch(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                grad_clip_norm=0.0,
                amp=train_cfg["amp"],
                log_interval=0,
                phase="test",
                epoch=None,
                logger=logger,
                show_progress=bool(logging_cfg.get("show_progress", True)),
            )

        if is_main_process(distributed_cfg.rank):
            summary = {
                "dataset": dataset_summary,
                "gpu": gpu_info,
                "distributed": {
                    "enabled": distributed_enabled(distributed_cfg.devices),
                    "world_size": distributed_cfg.world_size,
                    "devices": distributed_cfg.devices,
                    "backend": distributed_cfg.backend,
                },
                "backbone_load_info": unwrap_model(model).backbone.load_info,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "test_metrics": test_metrics,
                "history": history,
            }
            with (run_dir_path / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            logger.info("Training summary: %s", json.dumps(summary, ensure_ascii=False))
    finally:
        cleanup_process_group()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    devices = resolve_training_devices(config["train"])
    validate_cuda_devices(devices)

    run_dir = ensure_dir(
        resolve_path(config["paths"]["output_root"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    copy2(resolve_path(args.config), run_dir / "config.yaml")

    if distributed_enabled(devices):
        mp.spawn(
            run_worker,
            nprocs=len(devices),
            args=(str(resolve_path(args.config)), str(run_dir)),
            join=True,
        )
    else:
        run_worker(0, str(resolve_path(args.config)), str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
