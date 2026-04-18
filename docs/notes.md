# Notes

## 需求
修复 `train_fakeavceleb.py` 在服务器上加载 AV-HuBERT backbone 时，因为新版 fairseq 需要 `pos_conv_depth` 等新字段而导致的模型构建失败。

## 修改文件

- src/models/avhubert_backbone.py
- tests/test_avhubert_env.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 在 `src/models/avhubert_backbone.py` 中新增 `_merge_model_config_defaults()`，构建 backbone 配置时先合并 fairseq `Wav2Vec2Config` 默认值，再合并 `AVHubertConfig` 默认值，最后覆盖 checkpoint 里的旧配置。
- 对旧 AV-HuBERT checkpoint 缺失的新版 fairseq 字段补兼容默认值：`pos_conv_depth=1`、`conv_pos_batch_norm=False`，避免 `TransformerEncoder` 初始化时因 `None > 1` 直接崩溃。
- 保留原有 checkpoint 读取兼容逻辑，不改训练流程、不改模型参数语义，只修配置适配层。
- 在 `tests/test_avhubert_env.py` 中补充模型配置兼容分支测试，覆盖“旧 checkpoint 不含新版 fairseq 字段时自动回填”的行为。

## 验证

```bash
python -m py_compile src/models/avhubert_backbone.py tests/test_avhubert_env.py
python -m unittest tests.test_avhubert_env -v
```

结果：通过。语法检查通过；当前本地环境缺少 `torch`，`tests.test_avhubert_env` 的 4 个用例均被正确 skip，没有出现新的导入或语法错误。

## Git

- branch: `main`
- commit: `fix: backfill fairseq config defaults for avhubert`
