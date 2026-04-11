# Notes

## 需求
为 MAVOS-DD 英语小样本补齐专用配置和零参数 wrapper，方便数据迁移后直接复用现有 AV-HuBERT 训练链。

## 修改文件
- README.md
- configs/mavos_dd_english_small_preprocess.yaml
- configs/mavos_dd_english_small_classifier.yaml
- scripts/preprocess_mavos_dd_english_small.py
- scripts/cache_mavos_dd_english_small_audio_features.py
- scripts/train_mavos_dd_english_small.py
- scripts/plot_mavos_dd_english_small.py
- tests/test_mavos_dd_wrappers.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增 `configs/mavos_dd_english_small_preprocess.yaml` 与 `configs/mavos_dd_english_small_classifier.yaml`，将现有预处理、音频缓存、训练、绘图链路完整映射到 `mavos_dd_english_small` 路径。
- 新增四个零参数 wrapper：预处理、音频缓存、训练、绘图，减少手工传 `--config` 的操作。
- 保持你已经在服务器上使用的英语小样本 split / 下载脚本不变，不再改动其采样和下载逻辑。
- 这一步只做“路径和入口适配”，不重写 dataset / model / train 主逻辑。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mavos_dd_wrappers.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python scripts/train_mavos_dd_english_small.py --help
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python scripts/plot_mavos_dd_english_small.py --help
```

结果：通过；`test_mavos_dd_wrappers.py` 1 条用例通过，英语小样本训练和绘图 wrapper 入口可正常启动。

## Git
- branch: `main`
- commit: `git commit -m "feat: add mavos-dd english small wrappers"`
