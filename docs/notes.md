# Notes

## 需求
优化训练吞吐：新增离线音频特征缓存，避免 dataset 在 `__getitem__()` 内为每个样本反复执行 `ffmpeg + logfbank`。

## 修改文件
- README.md
- configs/avhubert_classifier.yaml
- scripts/cache_av1m_audio_features.py
- src/data/audio_features.py
- src/data/av1m_mouth_roi_dataset.py
- tests/test_audio_features.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增 `src/data/audio_features.py`，统一处理音频缓存路径映射、`ffmpeg` PCM 提取、logfbank 计算和 stack。
- 新增 `scripts/cache_av1m_audio_features.py`，离线遍历 train/val/test split 并把音频特征缓存到 `artifacts/.../audio_features/`。
- `AV1MMouthRoiDataset` 改成只读取缓存 `.npy` 音频特征；缓存缺失时直接报错，不再在 `__getitem__()` 里重复起 `ffmpeg`。
- 训练配置新增 `paths.audio_feature_root` 和 `paths.ffmpeg_path`。
- README 同步更新为“预处理后先跑音频缓存，再启动训练”。
- 新增 `test_audio_features.py`，覆盖缓存路径映射和 stack 行为。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_audio_features.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from src.data.audio_features import compute_logfbank_features
features = compute_logfbank_features(Path('dataset/AV-Deepfake1M/val/id00012/21Uxsk56VDQ/00002/real.mp4'), '/usr/bin/ffmpeg')
print(features.shape, features.dtype)
PY
```

结果：通过；`test_audio_features.py` 2 条用例通过，全部 `unittest` 共 25 条通过；真实样本可提取出 `shape=(1535, 26)` 的离线音频特征。

## Git
- branch: `main`
- commit: `git commit -m "feat: add cached audio feature pipeline"`
