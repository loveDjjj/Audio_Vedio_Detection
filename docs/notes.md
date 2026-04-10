# Notes

## 需求
将独立 YAML 预处理运行时扩展为主进程汇总进度、多卡设备分配版本。

## 修改文件
- README.md
- configs/avhubert_preprocess.yaml
- src/preprocess/mouth_roi.py
- src/preprocess/runtime.py
- tests/test_preprocess_runtime.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- `src/preprocess/runtime.py` 扩展为多卡 runtime：根据 YAML 中的 `devices` 和 `workers_per_device` 为 worker 分配 GPU，并设置每个 worker 的 `CUDA_VISIBLE_DEVICES` 和 CPU 线程上限。
- 主进程新增单个汇总进度条；worker 不再各自打印 `tqdm`，而是通过队列向主进程上报处理进度。
- 默认值按 `8 x 4090 + 88 vCPU` 机器设置为：`devices=[0,1,2,3,4,5,6,7]`、`workers_per_device=2`、`cpu_threads_per_worker=4`，总计 `16` 个 worker。
- 保留严格 CNN/GPU 检测约束：不回退 CPU HOG detector。
- 新增 `unittest` 回归用例，覆盖多卡 worker 分配、线程环境变量和 summary 聚合。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_preprocess_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from src.preprocess.runtime import build_worker_assignments, build_worker_environment
print(build_worker_assignments([0, 1, 2], 2))
print(build_worker_environment(3, 4))
PY
```

结果：通过；`test_preprocess_runtime.py` 3 条用例通过，全部 `unittest` 共 15 条通过，worker 设备轮转和线程环境变量输出符合预期。

## Git
- branch: `main`
- commit: `git commit -m "feat: add multi-gpu yaml-driven preprocess runtime"`
