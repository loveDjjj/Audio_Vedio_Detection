# Notes

## 需求
将当前训练代码改成由 YAML 控制的单卡/多卡 DDP 版本，并检查当前逻辑是否可运行。

## 修改文件
- README.md
- configs/avhubert_classifier.yaml
- scripts/train_avhubert_classifier.py
- src/train/engine.py
- src/train/runtime.py
- tests/test_train_distributed_runtime.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 训练配置改成从 `train.devices` 读取单卡或多卡设备列表；当设备列表长度大于 1 时，训练入口自动使用单机 DDP。
- `scripts/train_avhubert_classifier.py` 改成单机自拉起 worker 的多卡训练入口：主进程创建统一 `run_dir`，每个 worker 绑定自己的 GPU，自动初始化和清理 process group。
- `src/train/engine.py` 增加跨 rank 的 logits / targets / loss 聚合逻辑，保证多卡训练下指标、验证和测试只在主进程汇总。
- 新增 `src/train/runtime.py`，统一处理设备列表解析、rank 到 GPU 映射、主进程判定和 DDP 配置。
- checkpoint、history、summary 和终端日志都只由 rank 0 负责写出。
- 保持单卡模式可直接运行；默认配置仍是 `devices: [0]`，改成 `[0,1,...]` 即可切多卡。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_train_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_train_distributed_runtime.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
from src.train.runtime import build_distributed_config
cfg = build_distributed_config(
    {"devices": [0, 2], "backend": "nccl", "master_addr": "127.0.0.1", "master_port": 29501},
    local_rank=1,
)
print(cfg)
PY
```

结果：通过；`test_train_runtime.py` 4 条用例通过，`test_train_distributed_runtime.py` 5 条用例通过，全部 `unittest` 共 20 条通过；DDP 配置可正确将 `local_rank=1` 映射到设备 `2`。

## Git
- branch: `main`
- commit: `git commit -m "feat: add yaml-driven multi-gpu ddp training"`
