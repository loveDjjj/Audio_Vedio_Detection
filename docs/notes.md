# Notes

## 需求
为当前预处理和训练链路增加进度条显示与日志系统。

## 修改文件
- configs/avhubert_classifier.yaml
- configs/avhubert_preprocess.yaml
- scripts/train_avhubert_classifier.py
- src/preprocess/runtime.py
- src/train/engine.py
- src/utils/logging_utils.py
- tests/test_logging_utils.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增公共日志工具 `src/utils/logging_utils.py`，统一终端输出和文件日志格式，并使用 `tqdm.write` 避免破坏进度条。
- 训练配置新增 `logging` 区域；主进程将日志写到 `train.log`，其它 rank 写到 `train_rank{rank}.log`。
- 预处理配置新增 `logging` 区域；主进程将日志写到 `preprocess.log`，worker 写到 `preprocess_rank{rank}.log`。
- 训练 `run_epoch` 现在支持 phase 级进度条，主进程会显示 `train/val/test` 进度并定期记录 loss。
- 预处理 runtime 现在会把 worker 分配、配置摘要和最终 summary 写入日志文件，同时保留主进程总进度条。
- 新增 `test_logging_utils.py`，覆盖日志文件写盘能力。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_logging_utils.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
```

结果：通过；`test_logging_utils.py` 1 条用例通过，全部 `unittest` 共 23 条通过。

## Git
- branch: `main`
- commit: `git commit -m "feat: add progress bars and logging"`
