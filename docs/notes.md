# Notes

## 需求
实现 MAVOS-DD metadata 本地统计脚本，并安装 `datasets / pyarrow / pandas` 依赖。

## 修改文件
- README.md
- requirements.txt
- scripts/inspect_mavos_dd_metadata.py
- src/data/mavos_dd_metadata.py
- tests/test_mavos_dd_metadata.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 新增 `src/data/mavos_dd_metadata.py`，负责定位本地 `data-*.arrow`、读取记录并统计 split / language / generative_method / open-set 标记分布。
- 新增 `scripts/inspect_mavos_dd_metadata.py`，可直接对本地 `dataset/MAVOS-DD-meta` 导出 `mavos_dd_summary.json`。
- 在 `avhubert` 环境中安装 `datasets`、`pyarrow`、`pandas`，满足读取 Hugging Face Arrow metadata 的依赖。
- README 和 requirements 已同步补充 MAVOS-DD metadata 工具说明与依赖。
- 当前本地 `dataset/MAVOS-DD-meta` 目录里只有 `README.md`，metadata Arrow 文件尚未拉到本地，所以脚本已可用但还不能在这台机器上直接产出真实 MAVOS-DD 分布统计。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mavos_dd_metadata.py'
/root/shared-nvme/conda/envs/avhubert/bin/python -m pip show datasets pyarrow pandas
```

结果：通过；`test_mavos_dd_metadata.py` 1 条用例通过，且 `datasets / pyarrow / pandas` 已安装到 `avhubert` 环境。

## Git
- branch: `main`
- commit: `git commit -m "feat: add mavos-dd metadata inspector"`
