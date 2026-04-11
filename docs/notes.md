# Notes

## 需求
修复 MAVOS-DD 英语小样本音频缓存阶段在 `spawn + ProcessPoolExecutor` 下的多进程启动错误，使 `scripts/cache_mavos_dd_english_small_audio_features.py` 能正常启动并继续后续缓存。

## 修改文件
- src/data/audio_cache_runtime.py
- tests/test_audio_cache_runtime.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 为音频缓存 runtime 新增 `build_audio_cache_progress_queue()`，统一使用 `Manager().Queue()` 构造跨进程进度队列，避免普通 `Queue` 在 `spawn` 模式下被 `ProcessPoolExecutor` pickle 时抛出 `Queue objects should only be shared between processes through inheritance`。
- 将多进程音频缓存主流程切换到 manager-backed queue，并在结束时显式 `manager.shutdown()`。
- 新增回归测试，覆盖“必须使用 manager queue，而不是 plain context Queue”的约束。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_audio_cache_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mouth_roi.py'
```

结果：通过；`test_audio_cache_runtime.py` 4 条用例通过，`test_mouth_roi.py` 4 条用例通过。

## Git
- branch: `main`
- commit: `git commit -m "fix: use manager queue for audio cache workers"`
