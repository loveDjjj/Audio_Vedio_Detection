# Notes

## 需求
调整 `.gitignore`，让 `outputs/` 下的实验配置、结果摘要和训练曲线图可以被 Git 跟踪，同时继续忽略其它输出产物。

## 修改文件
- .gitignore
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 在 `.gitignore` 中改成 `/outputs/**` 递归忽略策略，同时递归放开 `outputs/` 下的目录访问权限。
- 新增仅针对 `outputs/` 的白名单规则：
  - `!/outputs/**/*.yaml`
  - `!/outputs/**/*.json`
  - `!/outputs/**/*.png`
- 这样像以下文件会被 Git 跟踪：
  - `outputs/avhubert/av1m_val_real_fullfake/20260411-040113/config.yaml`
  - `outputs/avhubert/av1m_val_real_fullfake/20260411-040113/summary.json`
  - `outputs/avhubert/av1m_val_real_fullfake/20260411-040113/training_curves.png`
- 其余如 `*.pt`、`*.log` 等输出文件仍保持忽略。

## 验证
```bash
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/best_head.pt
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/train.log
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/config.yaml || true
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/summary.json || true
git check-ignore -v outputs/avhubert/av1m_val_real_fullfake/20260411-040113/training_curves.png || true
```

结果：通过；`best_head.pt` 和 `train.log` 继续命中忽略规则，而 `config.yaml`、`summary.json`、`training_curves.png` 已命中对应的白名单放开规则。

## Git
- branch: `main`
- commit: `git commit -m "fix: track selected output artifacts"`
