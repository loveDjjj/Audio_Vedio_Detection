# Notes

## 需求

将三个数据集的 split 协议切换为当前仓库可支持的全量 `real/real vs fake/fake` 版本，并同步修正 AV1M 默认原始数据根路径与相关说明文档。

## 修改文件

- dataset/build_av1m_val_real_fullfake_splits.py
- dataset/build_fakeavceleb_real_fullfake_splits.py
- dataset/build_mavos_dd_english_splits.py
- src/data/fakeavceleb_subset.py
- src/data/mavos_dd_subset.py
- configs/avhubert_classifier.yaml
- configs/avhubert_preprocess.yaml
- README.md
- docs/full_dataset_runbook.md
- docs/notes.md
- docs/logs/2026-04.md
- tests/test_fakeavceleb_subset.py
- tests/test_av1m_fullfake_splits.py
- tests/test_mavos_dd_subset.py

## 修改内容

- AV1M：split builder 改为同时读取 `train_metadata.json` 与 `val_metadata.json`，保留全部 `real.mp4` / `fake_video_fake_audio.mp4`；官方 `train` 直接作为训练集，官方 `val` 按 clip 级随机切成 `val/test`；`relative_path` 增加 `train/`、`val/` 前缀以匹配后续预处理与训练链路。
- FakeAVCeleb：移除原先的 `1:1` fake 下采样，改为保留全部 `RealVideo-RealAudio` 与 `FakeVideo-FakeAudio` 样本，并按 `label/method` 分层切分 `train/val/test`。
- MAVOS-DD：移除 `english-only`、`open_set_model` 和比例采样限制，改为保留官方 `train / validation / test` 下全部 `real/real` 与 `fake/fake` 样本；脚本和配置文件名保留历史命名以兼容现有入口。
- 配置与文档：AV1M 默认 `raw_video_root` 改为 `/data/OneDay/AV-Deepfake1M`；README 与运行手册同步更新为新的全量协议说明，并补充 AV1M 仍需提前准备 `train/` 与 `train_metadata.json`。
- 测试：更新现有 FakeAVCeleb 单测，并新增 AV1M / MAVOS-DD 的全量协议验证用例用于本地回归。

## 验证

```bash
python -m unittest tests.test_av1m_fullfake_splits -v
python -m unittest tests.test_fakeavceleb_subset -v
python -m unittest tests.test_mavos_dd_subset -v
python dataset/build_av1m_val_real_fullfake_splits.py --help
python dataset/build_fakeavceleb_real_fullfake_splits.py --help
python dataset/build_mavos_dd_english_splits.py --help
python - <<'PY'
from pathlib import Path
from src.utils.project import load_config
for path in ("configs/avhubert_classifier.yaml", "configs/avhubert_preprocess.yaml"):
    cfg = load_config(Path(path))
    print(path, cfg["paths"]["raw_video_root"])
PY
```

结果：通过。三组针对性单测全部通过；三个 split builder 的 `--help` 均可正常启动；AV1M 训练与预处理配置解析出的 `raw_video_root` 均已切换到 `/data/OneDay/AV-Deepfake1M`。

## Git

- branch: `main`
- commit: 待确认
