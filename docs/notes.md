# Notes

## 需求
修复训练阶段 `avhubert.utils` 被错误解析为 `fairseq.utils` 导致的数据集初始化失败，并检查当前训练逻辑是否至少能读取真实样本。

## 修改文件
- src/data/av1m_mouth_roi_dataset.py
- tests/test_avhubert_dataset_import.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 将 `src/data/av1m_mouth_roi_dataset.py` 中的 `avhubert.utils` 导入改成显式子模块导入，避免被 `avhubert.__init__` 的星号导出污染成 `fairseq.utils`。
- 同时修正 transform 初始化仍引用旧局部变量名的问题。
- 音频提取由 `ffmpeg -> wav pipe` 改成直接读取 `s16le` PCM，避免训练时反复出现 `WavFileWarning: Reached EOF prematurely`。
- 新增最小回归测试，验证 `avhubert.utils` 子模块能够解析到真正的预处理变换实现。

## 验证
```bash
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

结果：通过；`test_avhubert_dataset_import.py` 1 条用例通过，全部 `unittest` 共 21 条通过；dataset 可成功初始化并读取首个真实样本，输出 `audio=(383, 104)`、`video=(383, 88, 88, 1)`。

## Git
- branch: `main`
- commit: `git commit -m "fix: resolve avhubert dataset utils import"`
