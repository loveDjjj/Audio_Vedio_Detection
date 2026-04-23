from __future__ import annotations

import csv
import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.audio_features import BASE_AUDIO_FEAT_DIM, load_cached_audio_feature_array, resolve_audio_feature_path
from src.utils.avhubert_env import bootstrap_avhubert_repo


CORRUPTED_SAMPLE_PLACEHOLDER_FRAMES = 8


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    label: int
    raw_video_path: Path
    mouth_roi_path: Path


class AV1MMouthRoiDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        raw_video_root: Path,
        mouth_roi_root: Path,
        audio_feature_root: Path,
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
        self.audio_feature_root = audio_feature_root
        self.avhubert_repo = avhubert_repo
        self.training = training
        self.image_crop_size = image_crop_size
        self.stack_order_audio = stack_order_audio
        self.normalize_audio = normalize_audio
        self.records: list[SampleRecord] = []
        self.missing_files = 0
        self.missing_mouth_roi_files = 0
        self.missing_raw_video_files = 0
        self.missing_audio_feature_files = 0

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
                audio_feature_path = resolve_audio_feature_path(audio_feature_root, relative_path)
                if not audio_feature_path.exists():
                    self.missing_audio_feature_files += 1
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state["avhubert_utils"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.__dict__.get("avhubert_utils") is None:
            bootstrap_avhubert_repo(self.__dict__["avhubert_repo"])
            self.avhubert_utils = importlib.import_module("avhubert.utils")

    def _load_cached_audio_features(self, relative_path: str) -> torch.Tensor:
        audio_feature_path = resolve_audio_feature_path(self.audio_feature_root, relative_path)
        if not audio_feature_path.exists():
            raise FileNotFoundError(
                f"Missing cached audio features: {audio_feature_path}. "
                "Run `python scripts/cache_av1m_audio_features.py` first."
            )

        audio_features = load_cached_audio_feature_array(audio_feature_path, self.stack_order_audio)
        audio = torch.from_numpy(audio_features.astype(np.float32))
        if self.normalize_audio:
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape[1:])
        return audio

    def _build_corrupted_sample_placeholder(self, record: SampleRecord) -> dict:
        return {
            "relative_path": record.relative_path,
            "label": record.label,
            "audio": torch.zeros(
                (CORRUPTED_SAMPLE_PLACEHOLDER_FRAMES, self.stack_order_audio * BASE_AUDIO_FEAT_DIM),
                dtype=torch.float32,
            ),
            "video": torch.zeros(
                (CORRUPTED_SAMPLE_PLACEHOLDER_FRAMES, self.image_crop_size, self.image_crop_size, 1),
                dtype=torch.float32,
            ),
            "sample_weight": 0.0,
        }

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        try:
            frames = self.avhubert_utils.load_video(str(record.mouth_roi_path)).astype(np.float32)
            frames = self.transform(frames).astype(np.float32)
            frames = np.expand_dims(frames, axis=-1)
            video = torch.from_numpy(frames)
            audio = self._load_cached_audio_features(record.relative_path)

            diff = audio.shape[0] - video.shape[0]
            if diff < 0:
                audio = torch.cat([audio, audio.new_zeros((-diff, audio.shape[1]))], dim=0)
            elif diff > 0:
                audio = audio[:-diff]
        except Exception as exc:
            warnings.warn(
                f"Skipping corrupted sample {record.relative_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._build_corrupted_sample_placeholder(record)

        return {
            "relative_path": record.relative_path,
            "label": record.label,
            "audio": audio,
            "video": video,
            "sample_weight": 1.0,
        }
