# Notes

## 本次需求

为 MAVOS-DD 增加一个“本地已存在视频”划分脚本：仍然先按 metadata 筛选官方 split 下的 `real/real` 与 `fake/fake`，但只把 `/data/OneDay/MAVOS-DD` 下真实存在的 mp4 写入新的 split，方便在完整数据尚未下载完成时先跑通预处理和训练链路。

## 实际修改文件

- dataset/build_mavos_dd_local_available_real_fullfake_splits.py
- src/data/mavos_dd_subset.py
- tests/test_mavos_dd_subset.py
- docs/notes.md
- docs/logs/2026-04.md

## 关键改动

- 新增 `build_local_available_real_fullfake_official_splits()`，复用现有 MAVOS-DD `real/real` vs `fake/fake` 官方 split 筛选逻辑，再根据 `raw_video_root / video_path` 的实际文件存在性过滤。
- 新增 `dataset/build_mavos_dd_local_available_real_fullfake_splits.py`，默认输出到 `splits/mavos_dd_real_fullfake_local_available`，避免覆盖完整官方协议的 `splits/mavos_dd_real_fullfake`。
- 新脚本的 `summary.json` 会记录 `availability`，包括 requested / available / missing 数量，以及按 split 和 label 统计的缺失情况。
- 补充单元测试，覆盖本地存在视频保留、缺失视频剔除、非 `real/real` / `fake/fake` 样本继续排除。

## 使用方式

```bash
python dataset/build_mavos_dd_local_available_real_fullfake_splits.py \
  --metadata-root /data/OneDay/MAVOS-DD \
  --raw-video-root /data/OneDay/MAVOS-DD \
  --output-dir splits/mavos_dd_real_fullfake_local_available
```

如需让现有 MAVOS-DD 配置使用该子集，需要把相关 YAML 的 `paths.split_dir` 以及 artifact/output 路径显式切到 local-available 名称；本次没有自动修改现有配置。

## 验证

```bash
python -m unittest tests.test_mavos_dd_subset -v
python -m py_compile src/data/mavos_dd_subset.py dataset/build_mavos_dd_local_available_real_fullfake_splits.py tests/test_mavos_dd_subset.py
python dataset/build_mavos_dd_local_available_real_fullfake_splits.py --help
```

结果：通过。

## 风险与限制

- `splits/mavos_dd_real_fullfake_local_available` 是本地可用子集，不等同于完整官方 MAVOS-DD real/fullfake 协议。
- 子集规模和类别分布取决于当前 `/data/OneDay/MAVOS-DD` 已下载文件；继续下载新视频后需要重新运行该脚本生成新的 split。
- 当前脚本只检查 mp4 文件是否存在，不校验视频是否可解码；损坏文件仍由后续 preprocess 阶段处理。

## Git

- branch: main
- commit: 待确认
