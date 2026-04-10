# Notes

## 需求
将当前训练入口改成 GPU-only，显式使用 CUDA 设备运行，不再静默回退到 CPU。

## 修改文件
- README.md
- configs/avhubert_classifier.yaml
- scripts/train_avhubert_classifier.py
- tests/test_train_runtime.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 训练脚本改成 GPU-only：`train.device` 必须是 CUDA 设备；当 CUDA 不可用时直接报错，而不是回退到 CPU。
- 默认设备改成 `cuda:0`，并在训练开始时显式打印正在使用的 GPU 名称和设备号。
- 启用 CUDA 运行时优化：`torch.cuda.set_device(...)`、`cudnn.benchmark=True`、TF32、`float32_matmul_precision=high`。
- DataLoader 的 `pin_memory` 和 `pin_memory_device` 跟随 CUDA 启用，减少 Host 到 GPU 的拷贝开销。
- 新增 `unittest` 回归用例，覆盖 CUDA 设备解析和 GPU-only 约束。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_train_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from scripts.train_avhubert_classifier import resolve_device, configure_cuda_runtime
device = resolve_device("cuda:0")
print(configure_cuda_runtime(device))
PY
```

结果：通过；`test_train_runtime.py` 4 条用例通过，全部 `unittest` 共 9 条通过，实际环境可解析到 `cuda:0` 并识别 `NVIDIA GeForce RTX 4090`。

## Git
- branch: `main`
- commit: `git commit -m "fix: enforce gpu-only avhubert training"`
