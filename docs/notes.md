# Notes

## 需求
进一步优化训练吞吐：将离线音频缓存脚本改成多进程版本，并直接缓存训练最终要读的 stacked 特征。

## 修改文件
- README.md
- configs/avhubert_classifier.yaml
- scripts/cache_av1m_audio_features.py
- src/data/audio_cache_runtime.py
- src/data/audio_features.py
- src/data/av1m_mouth_roi_dataset.py
- tests/test_audio_cache_runtime.py
- tests/test_audio_features.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增 `src/data/audio_cache_runtime.py`，负责多进程分片、主进程汇总进度条和日志输出。
- `scripts/cache_av1m_audio_features.py` 改成零业务参数的多进程缓存入口，默认读取 `configs/avhubert_classifier.yaml` 中的 `audio_cache` 配置。
- `src/data/audio_features.py` 继续负责 `ffmpeg` PCM 提取、logfbank 计算，并把 `stack_audio_features()` 作为缓存阶段的一部分。
- 音频缓存文件现在直接保存训练最终读取的 stacked 特征，避免训练时再次做 stack。
- `AV1MMouthRoiDataset` 改成只读取 stacked `.npy` 音频特征；缓存缺失时直接报错，不再在 `__getitem__()` 里重复起 `ffmpeg`。
- 训练配置新增 `audio_cache` 区块，默认值设为 `num_procs=16`、`cpu_threads_per_worker=4`、`stack_order_audio=4`。
- 新增 `test_audio_cache_runtime.py`，覆盖多进程分片和 summary 聚合逻辑。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_audio_cache_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_audio_features.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from pathlib import Path
from src.data.audio_features import compute_logfbank_features, stack_audio_features
features = compute_logfbank_features(Path('dataset/AV-Deepfake1M/val/id00012/21Uxsk56VDQ/00002/real.mp4'), '/usr/bin/ffmpeg')
stacked = stack_audio_features(features, 4)
print(features.shape, features.dtype)
print(stacked.shape, stacked.dtype)
PY
```

结果：通过；`test_audio_cache_runtime.py` 3 条用例通过，`test_audio_features.py` 2 条用例通过，全部 `unittest` 共 28 条通过；真实样本可提取出 `raw=(1535, 26)`、`stacked=(384, 104)` 的缓存特征。

## Git
- branch: `main`
- commit: `git commit -m "feat: add multi-process cached audio features"`
