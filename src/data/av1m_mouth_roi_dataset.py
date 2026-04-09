from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.avhubert_env import bootstrap_avhubert_repo


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    label: int


class AV1MMouthRoiDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        mouth_roi_root: Path,
        avhubert_repo: Path,
        training: bool,
        image_crop_size: int,
        image_mean: float,
        image_std: float,
        horizontal_flip_prob: float,
    ) -> None:
        super().__init__()
        bootstrap_avhubert_repo(avhubert_repo)
        from avhubert import utils as avhubert_utils  # type: ignore

        self.avhubert_utils = avhubert_utils
        self.csv_path = csv_path
        self.mouth_roi_root = mouth_roi_root
        self.training = training
        self.records: list[SampleRecord] = []
        self.missing_files = 0

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mouth_roi_path = mouth_roi_root / row["relative_path"]
                if not mouth_roi_path.exists():
                    self.missing_files += 1
                    continue
                self.records.append(
                    SampleRecord(
                        relative_path=row["relative_path"],
                        label=int(row["label"]),
                    )
                )

        transform_steps = [
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.RandomCrop((image_crop_size, image_crop_size))
            if training
            else avhubert_utils.CenterCrop((image_crop_size, image_crop_size)),
        ]
        if training:
            transform_steps.append(avhubert_utils.HorizontalFlip(horizontal_flip_prob))
        transform_steps.append(avhubert_utils.Normalize(image_mean, image_std))
        self.transform = avhubert_utils.Compose(transform_steps)

        if not self.records:
            raise ValueError(
                f"No usable mouth ROI videos found for {csv_path}. "
                f"Checked under {mouth_roi_root}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        video_path = self.mouth_roi_root / record.relative_path
        frames = self.avhubert_utils.load_video(str(video_path)).astype(np.float32)
        frames = self.transform(frames).astype(np.float32)
        frames = np.expand_dims(frames, axis=-1)
        video = torch.from_numpy(frames)
        return {
            "relative_path": record.relative_path,
            "label": record.label,
            "video": video,
        }
