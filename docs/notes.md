# Notes

## 需求
将 mouth ROI 预处理配置拆到独立 YAML，并改成默认 2 进程分片的多进程预处理，不再依赖终端参数传 stage / manifest / rank / nshard。

## 修改文件
- README.md
- configs/avhubert_preprocess.yaml
- scripts/preprocess_av1m_mouth_roi.py
- src/preprocess/mouth_roi.py
- src/preprocess/runtime.py
- tests/test_preprocess_runtime.py
- tests/test_mouth_roi.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增独立预处理配置文件 `configs/avhubert_preprocess.yaml`，将路径、预处理参数和运行时参数从训练 YAML 中分离。
- `scripts/preprocess_av1m_mouth_roi.py` 改成零业务参数入口，默认直接读取 `configs/avhubert_preprocess.yaml`。
- 新增 `src/preprocess/runtime.py`，负责 manifest 构建、worker 分片规划、`spawn` 多进程调度、per-shard summary 写出和最终 summary 聚合。
- 运行时默认使用 `2` 个进程分片，同一张 GPU 共享 CNN detector 阶段，避免直接把 4090 打爆；后续可在 YAML 内改为 `3` 或 `4`。
- 保留严格 CNN/GPU 检测约束：不回退 CPU HOG detector。
- 新增 `unittest` 回归用例，覆盖 worker shard 规划、summary 聚合和 CNN/GPU 检测约束。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_preprocess_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mouth_roi.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
import dlib
print(dlib.DLIB_USE_CUDA, dlib.cuda.get_num_devices())
PY
```

结果：通过；`test_preprocess_runtime.py` 2 条用例通过，`test_mouth_roi.py` 3 条用例通过，全部 `unittest` 共 14 条通过，当前环境中 `dlib.DLIB_USE_CUDA = True` 且 `dlib.cuda.get_num_devices() = 1`。

## Git
- branch: `main`
- commit: `git commit -m "feat: add yaml-driven multi-process mouth roi preprocessing"`
