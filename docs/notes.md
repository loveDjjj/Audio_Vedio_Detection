# Notes

## 需求

将三套数据集默认配置调整为适配 `2 x RTX 4090 48G + 72 CPU cores` 的服务器，并修正运行手册中的环境说明：`oneday` 用于 dlib 预处理，训练与其余脚本使用 `avhubert`。

## 修改文件

- configs/avhubert_classifier.yaml
- configs/avhubert_preprocess.yaml
- configs/fakeavceleb_classifier.yaml
- configs/fakeavceleb_preprocess.yaml
- configs/mavos_dd_real_fullfake_classifier.yaml
- configs/mavos_dd_real_fullfake_preprocess.yaml
- docs/full_dataset_runbook.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 训练配置：将 AV1M、FakeAVCeleb、MAVOS-DD 三套训练配置的 `train.devices` 统一改为 `[0,1]`，并将 `num_workers` 从 `16` 调整为 `8`。
- 音频缓存：将三套训练配置的 `audio_cache.num_procs` 从 `16` 调整为 `12`，保留 `cpu_threads_per_worker=4`。
- 预处理配置：将三套预处理配置的 `runtime.devices` 统一改为 `[0,1]`，并调整为 `workers_per_device=3`、`cpu_threads_per_worker=8`。
- 运行手册：将环境说明改为 `oneday` 仅用于 dlib mouth ROI 预处理，`avhubert` 用于 split 构建、数据下载、音频缓存、训练和绘图；同时补充当前默认硬件规格说明。

## 验证

```bash
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
    if "train" in cfg:
        print("  train.devices =", cfg["train"]["devices"])
        print("  train.num_workers =", cfg["train"]["num_workers"])
        print("  audio_cache.num_procs =", cfg["audio_cache"]["num_procs"])
        print("  audio_cache.cpu_threads_per_worker =", cfg["audio_cache"]["cpu_threads_per_worker"])
    else:
        print("  runtime.devices =", cfg["runtime"]["devices"])
        print("  runtime.workers_per_device =", cfg["runtime"]["workers_per_device"])
        print("  runtime.cpu_threads_per_worker =", cfg["runtime"]["cpu_threads_per_worker"])
PY
Get-Content docs/full_dataset_runbook.md
```

结果：通过。6 个配置文件都能正常解析，训练配置已统一为 `devices=[0,1]`、`num_workers=8`、`audio_cache.num_procs=12`，预处理配置已统一为 `devices=[0,1]`、`workers_per_device=3`、`cpu_threads_per_worker=8`；运行手册中的环境说明也已修正。

## Git

- branch: `main`
- commit: `chore: retune configs for 2x4090 host`
