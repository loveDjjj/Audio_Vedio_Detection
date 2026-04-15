# Notes

## 需求

接入 FakeAVCeleb 独立训练分支，只使用 `dataset/FakeAVCeleb/RealVideo-RealAudio` 与 `dataset/FakeAVCeleb/FakeVideo-FakeAudio`，新增 split builder、专用 YAML、wrapper 与 committed splits，并保持与当前 AV-HuBERT 流程兼容。

## 修改文件

- dataset/build_fakeavceleb_real_fullfake_splits.py
- src/data/fakeavceleb_subset.py
- configs/fakeavceleb_preprocess.yaml
- configs/fakeavceleb_classifier.yaml
- scripts/preprocess_fakeavceleb.py
- scripts/cache_fakeavceleb_audio_features.py
- scripts/train_fakeavceleb.py
- tests/test_fakeavceleb_subset.py
- tests/test_fakeavceleb_wrappers.py
- splits/fakeavceleb_real_fullfake/train.csv
- splits/fakeavceleb_real_fullfake/val.csv
- splits/fakeavceleb_real_fullfake/test.csv
- splits/fakeavceleb_real_fullfake/summary.json
- README.md
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容

- 新增 `src/data/fakeavceleb_subset.py`，集中处理 FakeAVCeleb metadata 解析、`RealVideo-RealAudio` / `FakeVideo-FakeAudio` 过滤、按生成方法的 `1:1` 平衡采样、随机视频切分和 CSV/summary 写出。
- 新增 `dataset/build_fakeavceleb_real_fullfake_splits.py`，生成 `splits/fakeavceleb_real_fullfake/{train,val,test}.csv` 与 `summary.json`，并基于当前本地数据生成了 1000 条平衡子集。
- 新增 FakeAVCeleb 专用 preprocess/classifier YAML 与三个零参数 wrapper，直接复用当前 mouth ROI、audio cache 和训练入口。
- 新增 `tests/test_fakeavceleb_subset.py` 与 `tests/test_fakeavceleb_wrappers.py`，覆盖 FakeAVCeleb 子集构建逻辑、配置路径和 wrapper 调用路径。
- 更新 `README.md`，补充 FakeAVCeleb 分支的结构、关键文件、运行命令、配置说明和输出目录。

## 验证

```bash
python -m unittest tests.test_fakeavceleb_subset tests.test_fakeavceleb_wrappers -v
python dataset/build_fakeavceleb_real_fullfake_splits.py --root dataset/FakeAVCeleb --output-dir splits/fakeavceleb_real_fullfake --seed 42
python scripts/train_fakeavceleb.py --help
python - <<'PY'
import csv
from pathlib import Path
for name in ["train", "val", "test"]:
    path = Path("splits/fakeavceleb_real_fullfake") / f"{name}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    print(name, len(rows), reader.fieldnames)
PY
```

结果：通过。单测通过；split builder 生成 `train=800 / val=100 / test=100`；训练 wrapper `--help` 正常；三份 CSV 均包含 `relative_path`、`label`、`method` 等预期字段。

## Git

- branch: `main`
- commit: `git commit -m "feat: add fakeavceleb real fullfake pipeline"`
