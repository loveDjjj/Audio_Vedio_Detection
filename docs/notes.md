# Notes

## 需求
实现英语子集的 MAVOS-DD 小样本 split 构建和按样本下载脚本，用于多生成器 pilot。

## 修改文件
- README.md
- dataset/build_mavos_dd_english_splits.py
- dataset/download_mavos_dd_selected_files.py
- src/data/mavos_dd_subset.py
- tests/test_mavos_dd_subset.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增 `dataset/build_mavos_dd_english_splits.py`，从本地 MAVOS-DD metadata 中构建英语小样本 `train.csv`、`val.csv`、`test.csv`。
- 采样策略固定为：
  - `train`: 英语 `train` split 按生成器分层抽样
  - `val`: 英语 `validation` 全量保留
  - `test`: 英语 `test` 且 `open_set_model=true`，按生成器分层抽样
- 新增 `dataset/download_mavos_dd_selected_files.py`，只下载上述 CSV 中出现的样本文件。
- 新增 `src/data/mavos_dd_subset.py` 和回归测试，保证分层抽样不会丢生成器覆盖。
- 用真实 metadata 做了 smoke check，英语 1/5 方案得到：
  - `train = 1277`
  - `val = 1079`
  - `test = 1591`

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mavos_dd_subset.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python dataset/build_mavos_dd_english_splits.py --metadata-root dataset/MAVOS-DD-meta --output-dir /tmp/mavos_dd_english_small_test --train-ratio 0.2 --test-ratio 0.2
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python dataset/download_mavos_dd_selected_files.py --help
```

结果：通过；`test_mavos_dd_subset.py` 2 条用例通过，英语 1/5 采样 split 成功生成，下载脚本入口可正常启动。

## Git
- branch: `main`
- commit: `git commit -m "feat: add mavos-dd english subset tools"`
