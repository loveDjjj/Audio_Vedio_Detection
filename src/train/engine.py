from __future__ import annotations

import torch
import torch.distributed as dist
from torch import amp as torch_amp
from tqdm import tqdm

from src.train.metrics import compute_binary_metrics


def format_batch_debug(batch: dict) -> dict:
    return {
        "audio_shape": tuple(batch["audio"].shape) if batch.get("audio") is not None else None,
        "video_shape": tuple(batch["video"].shape) if batch.get("video") is not None else None,
        "padding_mask_shape": tuple(batch["padding_mask"].shape) if batch.get("padding_mask") is not None else None,
        "labels_shape": tuple(batch["labels"].shape) if batch.get("labels") is not None else None,
        "relative_paths": list(batch.get("relative_paths", [])),
    }


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_distributed()) or dist.get_rank() == 0


def _gather_variable_1d_tensor(tensor: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
    if not _is_distributed():
        return [tensor.cpu()]

    world_size = dist.get_world_size()
    size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size)
    max_size = int(max(item.item() for item in sizes))

    padded = torch.zeros((max_size,), device=device, dtype=tensor.dtype)
    if tensor.numel() > 0:
        padded[: tensor.numel()] = tensor

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return [item[: int(size_tensor.item())].cpu() for item, size_tensor in zip(gathered, sizes)]


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    grad_clip_norm: float = 0.0,
    amp: bool = False,
    log_interval: int = 20,
    phase: str = "train",
    epoch: int | None = None,
    logger=None,
    show_progress: bool = False,
) -> dict[str, float] | None:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    iterator = loader
    progress_bar = None
    if show_progress and _is_main_process():
        desc = phase if epoch is None else f"{phase} e{epoch}"
        progress_bar = tqdm(loader, desc=desc, leave=False)
        iterator = progress_bar

    for step, batch in enumerate(iterator, start=1):
        if not batch:
            continue

        audio = batch["audio"]
        if audio is not None:
            audio = audio.to(device, non_blocking=True)
        video = batch["video"]
        if video is not None:
            video = video.to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        try:
            with torch_amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                video_logits, _frame_logits = model(audio=audio, video=video, padding_mask=padding_mask)
                loss = criterion(video_logits, targets)
        except RuntimeError:
            debug = format_batch_debug(batch)
            if device.type == "cuda":
                debug["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2)
                debug["cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 ** 2), 2)
            if logger is not None:
                logger.error("[batch-debug] %s", debug)
            else:
                print(f"[batch-debug] {debug}")
            raise

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        all_logits.append(video_logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if is_train and log_interval > 0 and step % log_interval == 0 and _is_main_process():
            if progress_bar is not None:
                progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")
            if logger is not None:
                logger.info("[%s] step=%s loss=%.4f", phase, step, loss.detach().item())
            else:
                print(f"[{phase}] step={step} loss={loss.detach().item():.4f}")

    if progress_bar is not None:
        progress_bar.close()

    if total_examples == 0 and not _is_distributed():
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0,), dtype=torch.float32)
    targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0,), dtype=torch.float32)

    loss_sum = torch.tensor([total_loss], device=device, dtype=torch.float64)
    example_count = torch.tensor([total_examples], device=device, dtype=torch.long)
    if _is_distributed():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(example_count, op=dist.ReduceOp.SUM)

    gathered_logits = _gather_variable_1d_tensor(logits.to(device), device=device)
    gathered_targets = _gather_variable_1d_tensor(targets.to(device), device=device)

    if not _is_main_process():
        return None

    global_examples = int(example_count.item())
    if global_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    merged_logits = torch.cat(gathered_logits, dim=0)
    merged_targets = torch.cat(gathered_targets, dim=0)
    metrics = compute_binary_metrics(merged_logits, merged_targets)
    metrics["loss"] = float(loss_sum.item()) / global_examples
    metrics["examples"] = float(global_examples)
    return metrics
