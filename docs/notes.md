# Notes

## 需求
修复当前单卡训练在 epoch 内部触发的 CUDA launch failure，并将默认训练配置调成更稳的保守值。

## 修改文件
- src/data/av1m_mouth_roi_dataset.py
- configs/avhubert_classifier.yaml
- scripts/train_avhubert_classifier.py
- src/train/engine.py
- tests/test_train_engine.py
- tests/test_avhubert_dataset_import.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 保留 `avhubert.utils` 显式子模块导入修复，并保留 `s16le` PCM 音频读取，确保 dataset 初始化和单样本读取稳定。
- 在 `src/train/engine.py` 中新增 batch 调试输出；一旦前向或 loss 计算阶段再次报 `RuntimeError`，会打印当前 batch 的 audio/video shape、relative_paths 和 CUDA 显存占用。
- 将训练默认配置调成更保守的单卡启动值：`data.max_frames=300`、`train.batch_size=2`、`train.num_workers=2`、`train.amp=false`。
- 关闭 `cudnn.benchmark`，避免变长序列 batch 频繁切换卷积算法时引入不稳定。
- 新增 `test_train_engine.py`，覆盖 batch 调试信息的格式化逻辑。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_train_engine.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_avhubert_dataset_import.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from pathlib import Path
from src.data.av1m_mouth_roi_dataset import AV1MMouthRoiDataset
from src.models.avhubert_backbone import load_avhubert_checkpoint_metadata
metadata = load_avhubert_checkpoint_metadata(Path('model/self_large_vox_433h.pt'))
dataset = AV1MMouthRoiDataset(
    csv_path=Path('splits/av1m_val_real_fullfake/train.csv'),
    raw_video_root=Path('dataset/AV-Deepfake1M/val'),
    mouth_roi_root=Path('artifacts/avhubert/av1m_val_real_fullfake/mouth_roi'),
    avhubert_repo=Path('third_party/av_hubert'),
    training=True,
    image_crop_size=88,
    image_mean=0.421,
    image_std=0.165,
    horizontal_flip_prob=0.5,
    stack_order_audio=metadata['stack_order_audio'],
    normalize_audio=metadata['audio_normalize'],
)
sample = dataset[0]
print(len(dataset), tuple(sample['audio'].shape), tuple(sample['video'].shape), sample['relative_path'])
PY
```

结果：通过；`test_train_engine.py` 1 条用例通过，`test_avhubert_dataset_import.py` 1 条用例通过，全部 `unittest` 共 22 条通过；dataset 可成功初始化并读取首个真实样本，输出 `audio=(383, 104)`、`video=(383, 88, 88, 1)`。

## Git
- branch: `main`
- commit: `git commit -m "fix: stabilize avhubert single-gpu training"`
