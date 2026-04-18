# Notes

## 需求

将三套数据集的 `artifacts` 路径统一迁移到 `/data/OneDay/artifacts`，并同步修改生成这些工件和消费这些工件的配置、测试与手册说明。

## 修改文件

- configs/avhubert_classifier.yaml
- configs/avhubert_preprocess.yaml
- configs/fakeavceleb_classifier.yaml
- configs/fakeavceleb_preprocess.yaml
- configs/mavos_dd_real_fullfake_classifier.yaml
- configs/mavos_dd_real_fullfake_preprocess.yaml
- tests/test_fakeavceleb_wrappers.py
- README.md
- docs/full_dataset_runbook.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- AV1M：将 `artifact_root`、`manifest_dir`、`landmark_dir`、`mouth_roi_root`、`audio_feature_root` 统一改到 `/data/OneDay/artifacts/avhubert/av1m_official_real_fullfake/...`。
- FakeAVCeleb：将同类工件路径统一改到 `/data/OneDay/artifacts/avhubert/fakeavceleb_real_fullfake/...`，并同步更新配置测试断言。
- MAVOS-DD：将同类工件路径统一改到 `/data/OneDay/artifacts/avhubert/mavos_dd_real_fullfake/...`。
- 文档：README 与运行手册中的 preprocess/audio cache 路径说明同步切换到 `/data/OneDay/artifacts/...`。

## 验证

```bash
python -m unittest tests.test_fakeavceleb_wrappers -v
python - <<'PY'
from pathlib import Path
from src.utils.project import load_config
for path in (
    "configs/avhubert_classifier.yaml",
    "configs/avhubert_preprocess.yaml",
    "configs/fakeavceleb_classifier.yaml",
    "configs/fakeavceleb_preprocess.yaml",
    "configs/mavos_dd_real_fullfake_classifier.yaml",
    "configs/mavos_dd_real_fullfake_preprocess.yaml",
):
    cfg = load_config(Path(path))
    print(path)
    for key in ("artifact_root", "manifest_dir", "landmark_dir", "mouth_roi_root"):
        print(" ", key, "=", cfg["paths"][key])
    if "audio_feature_root" in cfg["paths"]:
        print(" ", "audio_feature_root", "=", cfg["paths"]["audio_feature_root"])
PY
```

结果：通过。`tests.test_fakeavceleb_wrappers` 5 个用例全部通过；6 个配置文件解析出的 `artifact_root`、`manifest_dir`、`landmark_dir`、`mouth_roi_root` 和 `audio_feature_root` 都已切换到 `/data/OneDay/artifacts/avhubert/...`。

## Git

- branch: `main`
- commit: `chore: move artifacts to data disk`
