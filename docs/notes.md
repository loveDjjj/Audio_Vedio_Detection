# Notes

## 需求

为 FakeAVCeleb 设计独立训练分支，只接入 `dataset/FakeAVCeleb/RealVideo-RealAudio` 与 `dataset/FakeAVCeleb/FakeVideo-FakeAudio`，采用随机视频切分，并保持与当前 AV-HuBERT 流程兼容。

## 修改文件

- docs/superpowers/specs/2026-04-16-fakeavceleb-integration-design.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 新增 FakeAVCeleb 独立训练分支设计文档，明确数据范围、标签映射、1:1 平衡采样、随机视频切分、CSV 字段、配置文件与脚本增补方案。
- 记录本地 `meta_data.csv` 的尾部空列表头问题，并要求 split builder 使用显式列映射解析。
- 明确首版不引入 `RealVideo-FakeAudio`、`FakeVideo-RealAudio`、不混训，也不修改当前 trainer 的损失函数与采样器。

## 验证

```bash
Get-Content docs/superpowers/specs/2026-04-16-fakeavceleb-integration-design.md
```

结果：通过，设计文档已写入仓库，可用于后续评审与实现计划。

## Git

- branch: `main`
- commit: `git commit -m "docs: add fakeavceleb integration design spec"`
