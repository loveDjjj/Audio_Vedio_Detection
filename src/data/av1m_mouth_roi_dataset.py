from __future__ import annotations

import csv
import importlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from torch.utils.data import Dataset

from src.utils.avhubert_env import bootstrap_avhubert_repo


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    label: int
    raw_video_path: Path
    mouth_roi_path: Path


def _stack_audio_features(features: np.ndarray, stack_order: int) -> np.ndarray:
    if stack_order <= 1:
        return features

    feat_dim = features.shape[1]
    remainder = len(features) % stack_order
    if remainder:
        padding = np.zeros((stack_order - remainder, feat_dim), dtype=features.dtype)
        features = np.concatenate([features, padding], axis=0)
    return features.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)


class AV1MMouthRoiDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        raw_video_root: Path,
        mouth_roi_root: Path,
        avhubert_repo: Path,
        training: bool,
        image_crop_size: int,
        image_mean: float,
        image_std: float,
        horizontal_flip_prob: float,
        stack_order_audio: int,
        normalize_audio: bool,
    ) -> None:
        super().__init__()
        bootstrap_avhubert_repo(avhubert_repo)

        # `from avhubert import utils` is shadowed by `fairseq.utils` via AV-HuBERT's
        # wildcard imports, so import the preprocessing submodule explicitly.
        self.avhubert_utils = importlib.import_module("avhubert.utils")
        self.csv_path = csv_path
        self.raw_video_root = raw_video_root
        self.mouth_roi_root = mouth_roi_root
        self.training = training
        self.stack_order_audio = stack_order_audio
        self.normalize_audio = normalize_audio
        self.ffmpeg = shutil.which("ffmpeg")
        self.records: list[SampleRecord] = []
        self.missing_files = 0
        self.missing_mouth_roi_files = 0
        self.missing_raw_video_files = 0

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relative_path = row["relative_path"]
                mouth_roi_path = mouth_roi_root / relative_path
                raw_video_path = raw_video_root / relative_path
                if not mouth_roi_path.exists():
                    self.missing_mouth_roi_files += 1
                    self.missing_files += 1
                    continue
                if not raw_video_path.exists():
                    self.missing_raw_video_files += 1
                    self.missing_files += 1
                    continue
                self.records.append(
                    SampleRecord(
                        relative_path=relative_path,
                        label=int(row["label"]),
                        raw_video_path=raw_video_path,
                        mouth_roi_path=mouth_roi_path,
                    )
                )

        transform_steps = [
            self.avhubert_utils.Normalize(0.0, 255.0),
            self.avhubert_utils.RandomCrop((image_crop_size, image_crop_size))
            if training
            else self.avhubert_utils.CenterCrop((image_crop_size, image_crop_size)),
        ]
        if training:
            transform_steps.append(self.avhubert_utils.HorizontalFlip(horizontal_flip_prob))
        transform_steps.append(self.avhubert_utils.Normalize(image_mean, image_std))
        self.transform = self.avhubert_utils.Compose(transform_steps)

        if not self.records:
            raise ValueError(
                f"No usable AV samples found for {csv_path}. "
                f"Checked under {mouth_roi_root} and {raw_video_root}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _load_audio_features(self, raw_video_path: Path) -> torch.Tensor:
        if self.ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for SSR-DFD style audio+video AV-HuBERT training. "
                "Install ffmpeg first."
            )

        command = [
            self.ffmpeg,
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

        audio_features = logfbank(wav_data, samplerate=16_000).astype(np.float32)
        audio_features = _stack_audio_features(audio_features, self.stack_order_audio)
        audio = torch.from_numpy(audio_features.astype(np.float32))
        if self.normalize_audio:
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape[1:])
        return audio

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        frames = self.avhubert_utils.load_video(str(record.mouth_roi_path)).astype(np.float32)
        frames = self.transform(frames).astype(np.float32)
        frames = np.expand_dims(frames, axis=-1)
        video = torch.from_numpy(frames)
        audio = self._load_audio_features(record.raw_video_path)

        diff = audio.shape[0] - video.shape[0]
        if diff < 0:
            audio = torch.cat([audio, audio.new_zeros((-diff, audio.shape[1]))], dim=0)
        elif diff > 0:
            audio = audio[:-diff]

        return {
            "relative_path": record.relative_path,
            "label": record.label,
            "audio": audio,
            "video": video,
        }
