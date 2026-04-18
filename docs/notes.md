# Notes

## 需求

将仍保留历史命名的目录名和脚本名统一切换到新的全量协议命名，并删除旧 split 目录下的数据集 CSV/summary 工件。

## 修改文件

- dataset/build_av1m_official_real_fullfake_splits.py
- dataset/build_mavos_dd_real_fullfake_splits.py
- dataset/download_mavos_dd_selected_files.py
- configs/avhubert_classifier.yaml
- configs/avhubert_preprocess.yaml
- configs/mavos_dd_real_fullfake_classifier.yaml
- configs/mavos_dd_real_fullfake_preprocess.yaml
- scripts/preprocess_mavos_dd_real_fullfake.py
- scripts/cache_mavos_dd_real_fullfake_audio_features.py
- scripts/train_mavos_dd_real_fullfake.py
- scripts/plot_mavos_dd_real_fullfake.py
- splits/av1m_official_real_fullfake/train.csv
- splits/av1m_official_real_fullfake/val.csv
- splits/av1m_official_real_fullfake/test.csv
- splits/av1m_official_real_fullfake/summary.json
- splits/mavos_dd_real_fullfake/train.csv
- splits/mavos_dd_real_fullfake/val.csv
- splits/mavos_dd_real_fullfake/test.csv
- splits/mavos_dd_real_fullfake/summary.json
- README.md
- docs/full_dataset_runbook.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- AV1M：将 split builder 文件名改为 `build_av1m_official_real_fullfake_splits.py`，并将默认 `split_dir / artifact_root / output_root` 统一切换到 `av1m_official_real_fullfake`。
- MAVOS-DD：将 builder、配置和 wrapper 脚本从 `english_small` 命名统一改为 `real_fullfake`；同时把默认 `split_dir / artifact_root / output_root` 全部切到 `mavos_dd_real_fullfake`。
- split 工件：将已提交的 AV1M 与 MAVOS-DD `train.csv`、`val.csv`、`test.csv`、`summary.json` 迁移到新目录，并删除旧目录中的同名文件。
- 文档：README 和运行手册改为新的脚本名、目录名和命令，不再保留“历史命名沿用”的说明。

## 验证

```bash
python dataset/build_av1m_official_real_fullfake_splits.py --help
python dataset/build_mavos_dd_real_fullfake_splits.py --help
python scripts/train_mavos_dd_real_fullfake.py --help
python scripts/plot_mavos_dd_real_fullfake.py --help
python -m unittest tests.test_av1m_fullfake_splits tests.test_fakeavceleb_subset tests.test_mavos_dd_subset -v
python - <<'PY'
from pathlib import Path
from src.utils.project import load_config
for path in (
    "configs/avhubert_classifier.yaml",
    "configs/avhubert_preprocess.yaml",
    "configs/mavos_dd_real_fullfake_classifier.yaml",
    "configs/mavos_dd_real_fullfake_preprocess.yaml",
):
    cfg = load_config(Path(path))
    print(path, cfg["paths"]["split_dir"])
PY
```

结果：待验证。

## Git

- branch: `main`
- commit: `refactor: rename full-protocol dataset entrypoints`
