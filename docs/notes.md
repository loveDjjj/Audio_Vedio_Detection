# Notes

## 需求

新增一份全量数据集运行操作手册，按数据集逐个列出运行命令，并明确服务器上的环境切换约定：`avhubert` 只用于 dlib 预处理，`oneday` 用于其余步骤。

## 修改文件

- docs/full_dataset_runbook.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 新增 `docs/full_dataset_runbook.md`，按 AV-Deepfake1M、FakeAVCeleb、MAVOS-DD 三个分支分别给出 split 构建、预处理、音频缓存、训练、绘图的顺序命令。
- 在手册中明确环境切换规则：`conda activate avhubert` 仅用于 mouth ROI 预处理，`conda activate oneday` 用于其余脚本。
- 在手册中补充当前代码边界，明确哪些“全量”是当前仓库可以直接跑的范围，哪些还不是完整官方全量协议。
- 在手册中说明 `outputs/` 目录的实际产物包含模型权重，但 `.gitignore` 默认只跟踪 `yaml/json/png`。

## 验证

```bash
Get-Content docs/full_dataset_runbook.md
python dataset/build_av1m_val_real_fullfake_splits.py --help
python dataset/build_fakeavceleb_real_fullfake_splits.py --help
python dataset/build_mavos_dd_english_splits.py --help
python dataset/download_mavos_dd_selected_files.py --help
python scripts/inspect_mavos_dd_metadata.py --help
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/best_head.pt
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/summary.json
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/training_curves.png
```

结果：通过。`docs/full_dataset_runbook.md` 已生成；手册中引用的 5 个脚本入口均可正常显示 `--help`；`git check-ignore -v` 显示 `outputs/.../best_head.pt` 会被 ignore，而 `summary.json` 与 `training_curves.png` 不会被 ignore。

## Git

- branch: `main`
- commit: 待确认
