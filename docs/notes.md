# Notes

## 需求
按 SSR-DFD baseline 思路改造当前训练链：使用 frozen AV-HuBERT audio-visual backbone、单层线性 probe、frame logits + `logsumexp` 视频级聚合。

## 修改文件
- README.md
- configs/avhubert_classifier.yaml
- requirements.txt
- src/data/av1m_mouth_roi_dataset.py
- src/data/collate.py
- src/models/avhubert_backbone.py
- src/models/binary_detector.py
- src/train/engine.py
- scripts/train_avhubert_classifier.py
- src/utils/avhubert_env.py
- tests/test_avhubert_checkpoint.py
- tests/test_binary_detector.py
- tests/test_collate.py
- tests/test_avhubert_env.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 将当前 video-only `AV-HuBERT` 训练链改成 audio+video 版本：dataset 从 raw mp4 提取 16kHz mono 音频并计算 logfbank，同时读取 mouth ROI 视频。
- 新增 audio+video batch collate，统一对齐时序长度后输出 `audio: [B, F, T]`、`video: [B, C, T, H, W]` 和共享 `padding_mask`。
- 重写 backbone loader，直接从 checkpoint 的 backbone 配置和 state dict 构造 `AVHubertModel`，避免 seq2seq wrapper 对上游 label dictionary 路径的依赖。
- 将 detector 改成 SSR-DFD 风格 linear probe：对 `[B, T, 1024]` 时序特征做共享 `Linear(1024, 1)`，保留 `frame_logits`，并对有效时间步做 `logsumexp` 得到视频级 logit。
- 更新训练脚本和 engine，只优化线性层参数，并从 checkpoint 元数据读取 `stack_order_audio` 和音频归一化设置。
- 更新 README、requirements 注释和主配置说明，使仓库描述与新的 audio-visual baseline 保持一致。
- 新增 `unittest` 回归用例，覆盖 checkpoint 元数据解析、linear probe `logsumexp` 聚合逻辑和 audio+video collate 结果。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from pathlib import Path
from src.models.binary_detector import AVHubertBinaryDetector
model = AVHubertBinaryDetector(
    checkpoint_path=Path('model/self_large_vox_433h.pt'),
    avhubert_repo=Path('third_party/av_hubert'),
    freeze_backbone=True,
    feat_dim=1024,
)
print(type(model.backbone.model).__name__, model.backbone.output_dim, model.backbone.metadata['stack_order_audio'])
PY
```

结果：通过；`unittest` 5 条用例通过，smoke check 可实例化 frozen AV-HuBERT linear probe，输出维度为 `1024`，音频 `stack_order_audio` 为 `4`。

## Git
- branch: `main`
- commit: 待确认
