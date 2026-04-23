# Notes

## 本次需求
为预处理、音频缓存和训练阶段增加“损坏即跳过”的容错机制，避免运行过程中因为损坏的缓存工件直接中断。直接触发问题的是 `align` 阶段读取被截断的 `landmarks.pkl` 时崩溃；本次要求也覆盖后续的音频缓存和训练读取阶段。

## 实际修改文件

- src/preprocess/mouth_roi.py
- src/preprocess/runtime.py
- src/data/audio_features.py
- src/data/audio_cache_runtime.py
- src/data/av1m_mouth_roi_dataset.py
- src/data/collate.py
- src/train/engine.py
- scripts/train_avhubert_classifier.py
- tests/test_corruption_resilience.py
- docs/notes.md
- docs/logs/2026-04.md

## 关键改动

- 为 `align` 阶段补充损坏 `landmarks/*.pkl` 的保护：`pickle.load()` 失败时记入 `failed_read_landmarks`，并继续处理后续样本。
- 新增统一的缓存音频特征校验逻辑，要求已有 `.npy` 文件能够正常解码且维度符合当前堆叠规则，才视为可用缓存。
- 调整音频缓存阶段：遇到已存在但损坏的 `.npy` 不再盲目跳过，而是记为 `failed_invalid_existing_features` 并继续。
- 调整训练数据集读取逻辑：坏的 `mouth_roi` 或坏的音频特征不再在 `__getitem__()` 中直接抛异常，而是转换为零权重占位样本。
- 调整 `collate` 和训练 `engine`：忽略零权重损坏样本，但仍保证 DDP 场景下 batch 结构稳定，不因整批局部坏样本导致不同步。
- 将训练损失改为 `BCEWithLogitsLoss(reduction="none")`，以便按样本权重将损坏占位样本的贡献清零。
- 新增回归测试，覆盖损坏 landmark 跳过、损坏音频缓存跳过、占位样本生成和零权重 batch 处理。

## 验证

```bash
python -m py_compile src/data/audio_features.py src/data/audio_cache_runtime.py src/data/av1m_mouth_roi_dataset.py src/data/collate.py src/preprocess/mouth_roi.py src/preprocess/runtime.py src/train/engine.py scripts/train_avhubert_classifier.py tests/test_corruption_resilience.py

python -m unittest tests.test_fakeavceleb_wrappers tests.test_corruption_resilience -v
```

结果：通过。语法检查通过；单元测试通过。当前本地环境中有 3 个预期 skip，因为缺少 `python_speech_features` 和 `torch`。

## Git

- branch: `main`
- commit: `fix: skip corrupted cached preprocess and training artifacts`
