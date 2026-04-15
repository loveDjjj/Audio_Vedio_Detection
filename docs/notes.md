# Notes

## 需求

根据已批准的 FakeAVCeleb 接入设计，编写可执行的实现计划，覆盖 split builder、FakeAVCeleb 专用 YAML、wrapper、生成的 split 工件、测试与文档更新。

## 修改文件

- docs/superpowers/plans/2026-04-16-fakeavceleb-independent-training-branch.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 新增 FakeAVCeleb 独立训练分支实现计划文档，拆分为 split builder、配置与 wrapper、生成 split 与文档更新三组任务。
- 在计划中明确将使用 `src/data/fakeavceleb_subset.py` 作为可测试的元数据解析与采样模块，并新增 `tests/test_fakeavceleb_subset.py` 与 `tests/test_fakeavceleb_wrappers.py`。
- 同步更新当前 notes 与 2026-04 月志，记录本次计划文档产出与后续执行入口。

## 验证

```bash
Get-Content docs/superpowers/plans/2026-04-16-fakeavceleb-independent-training-branch.md
```

结果：通过，实现计划文档已写入仓库，可直接进入执行阶段。

## Git

- branch: `main`
- commit: `git commit -m "docs: add fakeavceleb implementation plan"`
