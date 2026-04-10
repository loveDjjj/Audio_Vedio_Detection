from __future__ import annotations

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

from src.train.metrics import compute_binary_metrics


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
) -> dict[str, float] | None:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for step, batch in enumerate(loader, start=1):
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

        with autocast(enabled=amp and device.type == "cuda"):
            video_logits, _frame_logits = model(audio=audio, video=video, padding_mask=padding_mask)
            loss = criterion(video_logits, targets)

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
            print(f"[train] step={step} loss={loss.detach().item():.4f}")

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
