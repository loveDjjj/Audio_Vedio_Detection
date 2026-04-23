from __future__ import annotations

import torch


def collate_audio_video_batch(samples: list[dict], max_frames: int, pad_to_batch_max: bool) -> dict:
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return {}

    valid_samples = [sample for sample in samples if float(sample.get("sample_weight", 1.0)) > 0.0]
    if valid_samples:
        samples = valid_samples
    else:
        samples = samples[:1]

    lengths: list[int] = []
    for sample in samples:
        sample_lengths = []
        if sample.get("audio") is not None:
            sample_lengths.append(sample["audio"].shape[0])
        if sample.get("video") is not None:
            sample_lengths.append(sample["video"].shape[0])
        if not sample_lengths:
            continue
        lengths.append(min(sample_lengths))

    if not lengths:
        return {}

    if pad_to_batch_max:
        target_frames = min(max(lengths), max_frames)
    else:
        target_frames = min(min(lengths), max_frames)

    batch_size = len(samples)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    sample_weights = torch.zeros((batch_size,), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, target_frames), dtype=torch.bool)
    relative_paths: list[str] = []

    has_audio = samples[0].get("audio") is not None
    has_video = samples[0].get("video") is not None

    audios = None
    videos = None
    if has_audio:
        feat_dim = samples[0]["audio"].shape[1]
        audios = samples[0]["audio"].new_zeros((batch_size, target_frames, feat_dim))
    if has_video:
        height, width, channels = samples[0]["video"].shape[1:]
        videos = samples[0]["video"].new_zeros((batch_size, target_frames, height, width, channels))

    for index, sample in enumerate(samples):
        usable_frames = target_frames
        if sample.get("audio") is not None:
            usable_frames = min(usable_frames, sample["audio"].shape[0])
        if sample.get("video") is not None:
            usable_frames = min(usable_frames, sample["video"].shape[0])

        if audios is not None:
            audios[index, :usable_frames] = sample["audio"][:usable_frames]
        if videos is not None:
            videos[index, :usable_frames] = sample["video"][:usable_frames]
        padding_mask[index, :usable_frames] = False
        labels[index] = float(sample["label"])
        sample_weights[index] = float(sample.get("sample_weight", 1.0))
        relative_paths.append(sample["relative_path"])

    if audios is not None:
        audios = audios.transpose(1, 2).contiguous()
    if videos is not None:
        videos = videos.permute(0, 4, 1, 2, 3).contiguous()

    return {
        "audio": audios,
        "video": videos,
        "padding_mask": padding_mask,
        "labels": labels,
        "sample_weights": sample_weights,
        "relative_paths": relative_paths,
    }


def collate_video_batch(samples: list[dict], max_frames: int, pad_to_batch_max: bool) -> dict:
    return collate_audio_video_batch(samples=samples, max_frames=max_frames, pad_to_batch_max=pad_to_batch_max)
