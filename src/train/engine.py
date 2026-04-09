from __future__ import annotations

import torch
from torch.cuda.amp import autocast

from src.train.metrics import compute_binary_metrics


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
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for step, batch in enumerate(loader, start=1):
        if not batch:
            continue

        video = batch["video"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        with autocast(enabled=amp and device.type == "cuda"):
            logits = model(video=video, padding_mask=padding_mask)
            loss = criterion(logits, targets)

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
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if is_train and log_interval > 0 and step % log_interval == 0:
            print(f"[train] step={step} loss={loss.detach().item():.4f}")

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_binary_metrics(logits, targets)
    metrics["loss"] = total_loss / total_examples
    metrics["examples"] = float(total_examples)
    return metrics
