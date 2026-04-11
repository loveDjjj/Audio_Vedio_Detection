# Notes

## 需求
增加可单独调用的训练可视化模块，并在每个 epoch 结束后自动刷新曲线图。

## 修改文件
- README.md
- configs/avhubert_classifier.yaml
- scripts/plot_training_summary.py
- scripts/train_avhubert_classifier.py
- src/visualization/training_curves.py
- tests/test_training_curves.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增 `src/visualization/training_curves.py`，统一读取 `summary.json` 的 `history` 并绘制 train/val 曲线图。
- 新增 `scripts/plot_training_summary.py`，支持直接对任意历史 `summary.json` 单独生成 `training_curves.png`，不需要重跑训练。
- 训练配置新增 `visualization` 区块；训练时 rank 0 会在每个 epoch 结束后更新一次 `training_curves.png`，训练结束后再用完整 `summary.json` 覆盖生成一次。
- 当前默认图包含 4 个子图：`loss`、`accuracy`、`f1`、`precision/recall`。
- 已用现有训练结果 [`20260411-040113/summary.json`](/root/shared-nvme/Audio_Vedio_Detection/outputs/avhubert/av1m_val_real_fullfake/20260411-040113/summary.json) 成功离线生成 [`training_curves.png`](/root/shared-nvme/Audio_Vedio_Detection/outputs/avhubert/av1m_val_real_fullfake/20260411-040113/training_curves.png)。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_training_curves.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib /root/shared-nvme/conda/envs/avhubert/bin/python scripts/plot_training_summary.py --summary outputs/avhubert/av1m_val_real_fullfake/20260411-040113/summary.json
```

结果：通过；`test_training_curves.py` 2 条用例通过，全部 `unittest` 共 31 条通过；离线绘图成功生成 `training_curves.png`。

## Git
- branch: `main`
- commit: `git commit -m "feat: add training visualization module"`
