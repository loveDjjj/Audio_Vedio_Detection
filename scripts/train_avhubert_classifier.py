from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from shutil import copy2

import torch
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.av1m_mouth_roi_dataset import AV1MMouthRoiDataset
from src.data.collate import collate_audio_video_batch
from src.models.avhubert_backbone import load_avhubert_checkpoint_metadata
from src.models.binary_detector import AVHubertBinaryDetector
from src.train.engine import run_epoch
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


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, max_frames: int, pad_to_batch_max: bool, drop_last: bool):
    pin_memory = torch.cuda.is_available()
    collate_fn = partial(
        collate_audio_video_batch,
        max_frames=max_frames,
        pad_to_batch_max=pad_to_batch_max,
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    if pin_memory:
        loader_kwargs["pin_memory_device"] = "cuda"
    return DataLoader(
        **loader_kwargs,
    )


def resolve_device(device_name: str) -> torch.device:
    if not device_name.startswith("cuda"):
        raise ValueError("This training entrypoint is GPU-only. Set `train.device` to a CUDA device such as `cuda:0`.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entrypoint, but `torch.cuda.is_available()` is false.")
    if device_name == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_name)


def configure_cuda_runtime(device: torch.device) -> dict[str, str | int]:
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_index": int(device.index or 0),
    }


def save_head_checkpoint(path: Path, epoch: int, model: AVHubertBinaryDetector, metrics: dict, config: dict) -> None:
    state = {
        "epoch": epoch,
        "head_state_dict": model.head.state_dict(),
        "metrics": metrics,
        "config": config,
        "backbone_load_info": model.backbone.load_info,
    }
    torch.save(state, path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["train"]["seed"])

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    paths = config["paths"]
    checkpoint_path = resolve_path(paths["checkpoint_path"])
    backbone_metadata = load_avhubert_checkpoint_metadata(checkpoint_path)

    if not model_cfg["freeze_backbone"]:
        raise ValueError("SSR-DFD AV-HuBERT baseline requires `model.freeze_backbone: true`.")

    device = resolve_device(train_cfg["device"])
    gpu_info = configure_cuda_runtime(device)
    print(f"Using GPU {gpu_info['gpu_index']}: {gpu_info['gpu_name']} ({gpu_info['device']})")
    run_dir = ensure_dir(
        resolve_path(paths["output_root"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    copy2(resolve_path(args.config), run_dir / "config.yaml")

    train_dataset = build_dataset("train", config, training=True, backbone_metadata=backbone_metadata)
    val_dataset = build_dataset("val", config, training=False, backbone_metadata=backbone_metadata)
    test_dataset = build_dataset("test", config, training=False, backbone_metadata=backbone_metadata)

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=data_cfg["shuffle_train"],
        max_frames=data_cfg["max_frames"],
        pad_to_batch_max=data_cfg["pad_to_batch_max"],
        drop_last=data_cfg["drop_last_train"],
    )
    eval_loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "shuffle": False,
        "max_frames": data_cfg["max_frames"],
        "pad_to_batch_max": data_cfg["pad_to_batch_max"],
        "drop_last": False,
    }
    val_loader = make_loader(val_dataset, **eval_loader_kwargs)
    test_loader = make_loader(test_dataset, **eval_loader_kwargs)

    model = AVHubertBinaryDetector(
        checkpoint_path=checkpoint_path,
        avhubert_repo=resolve_path(paths["avhubert_repo"]),
        freeze_backbone=model_cfg["freeze_backbone"],
        feat_dim=backbone_metadata["encoder_embed_dim"],
    ).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = BCEWithLogitsLoss()
    scaler = GradScaler(enabled=train_cfg["amp"])

    history: list[dict] = []
    best_epoch = -1
    best_val_f1 = float("-inf")
    best_path = run_dir / "best_head.pt"
    last_path = run_dir / "last_head.pt"

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

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"Epoch {epoch}/{train_cfg['epochs']}")
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
            )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2, ensure_ascii=False))

        save_head_checkpoint(last_path, epoch=epoch, model=model, metrics=val_metrics, config=config)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            save_head_checkpoint(best_path, epoch=epoch, model=model, metrics=val_metrics, config=config)

    if best_path.exists():
        best_state = torch.load(best_path, map_location="cpu")
        model.head.load_state_dict(best_state["head_state_dict"])

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
        )

    summary = {
        "dataset": dataset_summary,
        "gpu": gpu_info,
        "backbone_load_info": model.backbone.load_info,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
