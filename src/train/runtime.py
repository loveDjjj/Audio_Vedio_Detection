from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedConfig:
    devices: list[int]
    local_rank: int
    rank: int
    world_size: int
    device_index: int
    backend: str
    master_addr: str
    master_port: int


def resolve_training_devices(train_cfg: dict[str, Any]) -> list[int]:
    if "devices" in train_cfg:
        devices = [int(device) for device in train_cfg["devices"]]
        if not devices:
            raise ValueError("train.devices must contain at least one CUDA device index.")
        return devices

    device_name = train_cfg.get("device")
    if not device_name:
        raise ValueError("Training config must provide either `train.devices` or `train.device`.")
    if not str(device_name).startswith("cuda"):
        raise ValueError("Training entrypoint is GPU-only; use CUDA devices.")
    if ":" in str(device_name):
        return [int(str(device_name).split(":", 1)[1])]
    return [0]


def distributed_enabled(devices: list[int]) -> bool:
    return len(devices) > 1


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_distributed_config(train_cfg: dict[str, Any], local_rank: int, rank: int | None = None) -> DistributedConfig:
    devices = resolve_training_devices(train_cfg)
    world_size = len(devices)
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"local_rank must be in [0, {world_size - 1}], got {local_rank}")

    return DistributedConfig(
        devices=devices,
        local_rank=local_rank,
        rank=local_rank if rank is None else rank,
        world_size=world_size,
        device_index=devices[local_rank],
        backend=str(train_cfg.get("backend", "nccl")),
        master_addr=str(train_cfg.get("master_addr", "127.0.0.1")),
        master_port=int(train_cfg.get("master_port", 29501)),
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
