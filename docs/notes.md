# Notes

## 需求
为 `docs/avhubert_baseline_experiment_analysis.md` 中的 4 张训练曲线图片补充简短图注，说明对应数据集、关键训练参数和图像含义。

## 修改文件
- docs/avhubert_baseline_experiment_analysis.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 在实验结果部分的 4 张训练曲线图下方新增图注。
- 图注统一补充：
  - 数据集/实验名称
  - `epoch`
  - `batch_size`
  - `8 GPUs`
  - 该图对应的简要作用说明
- 未修改文档整体结构，只增强图片解释性。

## 验证
```bash
grep -n "图 1：\\|图 2：\\|图 3：\\|图 4：" docs/avhubert_baseline_experiment_analysis.md
```

结果：通过；4 条图注已写入文档。

## Git
- branch: `main`
- commit: `git commit -m "docs: add captions to avhubert experiment figures"`
