from __future__ import annotations

import torch


def collate_video_batch(samples: list[dict], max_frames: int, pad_to_batch_max: bool) -> dict:
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return {}

    lengths = [sample["video"].shape[0] for sample in samples]
    if pad_to_batch_max:
        target_frames = min(max(lengths), max_frames)
    else:
        target_frames = min(min(lengths), max_frames)

    batch_size = len(samples)
    height, width, channels = samples[0]["video"].shape[1:]
    videos = samples[0]["video"].new_zeros((batch_size, target_frames, height, width, channels))
    padding_mask = torch.ones((batch_size, target_frames), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    relative_paths: list[str] = []

    for index, sample in enumerate(samples):
        video = sample["video"]
        usable_frames = min(video.shape[0], target_frames)
        videos[index, :usable_frames] = video[:usable_frames]
        padding_mask[index, :usable_frames] = False
        labels[index] = float(sample["label"])
        relative_paths.append(sample["relative_path"])

    videos = videos.permute(0, 4, 1, 2, 3).contiguous()
    return {
        "video": videos,
        "padding_mask": padding_mask,
        "labels": labels,
        "relative_paths": relative_paths,
    }
