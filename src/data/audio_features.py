from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from python_speech_features import logfbank


BASE_AUDIO_FEAT_DIM = 26


def resolve_audio_feature_path(audio_feature_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.suffix != ".mp4":
        raise ValueError(f"Expected an .mp4 relative path, got: {relative_path}")
    return audio_feature_root / path.with_suffix(".npy")


def resolve_ffmpeg_binary(configured_path: str | Path | None) -> str:
    candidates: list[str] = []
    if configured_path:
        candidates.append(str(configured_path))
    detected = shutil.which("ffmpeg")
    if detected:
        candidates.append(detected)

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Unable to find `ffmpeg`. Set `paths.ffmpeg_path` to a valid executable or install ffmpeg into PATH."
    )


def extract_pcm16_audio_from_video(raw_video_path: Path, ffmpeg: str) -> np.ndarray:
    command = [
        ffmpeg,
        "-nostdin",
        "-i",
        str(raw_video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed for {raw_video_path}: {error}")

    wav_data = np.frombuffer(result.stdout, dtype=np.int16)
    if wav_data.ndim != 1 or wav_data.size == 0:
        raise ValueError(f"Expected non-empty 16kHz mono PCM audio from {raw_video_path}.")
    return wav_data


def compute_logfbank_features(raw_video_path: Path, ffmpeg: str) -> np.ndarray:
    wav_data = extract_pcm16_audio_from_video(raw_video_path=raw_video_path, ffmpeg=ffmpeg)
    return logfbank(wav_data, samplerate=16_000).astype(np.float32)


def load_cached_audio_feature_array(feature_path: Path, stack_order: int) -> np.ndarray:
    audio_features = np.load(feature_path, allow_pickle=False).astype(np.float32, copy=False)
    if audio_features.ndim != 2:
        raise ValueError(f"Expected 2D cached audio features, got shape={audio_features.shape} from {feature_path}")

    expected_dim = stack_order * BASE_AUDIO_FEAT_DIM
    if audio_features.shape[1] == expected_dim:
        return audio_features
    if audio_features.shape[1] == BASE_AUDIO_FEAT_DIM:
        return stack_audio_features(audio_features, stack_order)
    raise ValueError(
        f"Expected cached audio feature dim {BASE_AUDIO_FEAT_DIM} or {expected_dim}, "
        f"got shape={audio_features.shape} from {feature_path}"
    )


def stack_audio_features(features: np.ndarray, stack_order: int) -> np.ndarray:
    if stack_order <= 1:
        return features

    feat_dim = features.shape[1]
    remainder = len(features) % stack_order
    if remainder:
        padding = np.zeros((stack_order - remainder, feat_dim), dtype=features.dtype)
        features = np.concatenate([features, padding], axis=0)
    return features.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
