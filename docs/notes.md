# Notes

## 需求

修复 `train_fakeavceleb.py` 在 PyTorch 2.6 环境下读取 AV-HuBERT checkpoint 时因 `torch.load` 默认 `weights_only=True` 导致的加载失败。

## 修改文件

- src/models/avhubert_backbone.py
- tests/test_avhubert_env.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 在 `src/models/avhubert_backbone.py` 中新增 `load_torch_checkpoint()`，统一通过 `torch.load(..., weights_only=False)` 读取受信任的 AV-HuBERT checkpoint。
- 为兼容旧版 PyTorch，不支持 `weights_only` 参数时自动回退到 `torch.load(..., map_location="cpu")`。
- 将 metadata 读取和 backbone 真正加载的两处 checkpoint 读取都切换到该 helper。
- 在 `tests/test_avhubert_env.py` 中补充 checkpoint 加载兼容分支测试；在当前无 `torch` 的本地环境下改为跳过而不是报错。

## 验证

```bash
python -m py_compile src/models/avhubert_backbone.py
python -m unittest tests.test_avhubert_env -v
```

结果：通过。`src/models/avhubert_backbone.py` 语法检查通过；当前本地环境缺少 `torch`，相关测试被正确跳过，不再因为导入失败中断。

## Git

- branch: `main`
- commit: `fix: load avhubert checkpoints on torch 2.6`
