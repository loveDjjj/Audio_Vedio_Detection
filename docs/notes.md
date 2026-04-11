# Notes

## 需求
修复 MAVOS-DD 英语小样本预处理中“单个坏视频会打崩整轮”的问题，并将已确认损坏的样本从当前 `test` split 中移除，避免后续预处理、音频缓存和训练再次纳入。

## 修改文件
- src/preprocess/mouth_roi.py
- src/preprocess/runtime.py
- splits/mavos_dd_english_small/test.csv
- splits/mavos_dd_english_small/summary.json
- tests/test_mouth_roi.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 在 `process_manifest()` 中为检测阶段补上坏视频保护：当 `detect_landmarks_for_video()` 因损坏 mp4 无法读帧时报错时，记录为 `failed_read_video` 并继续处理其它样本，不再让整个多进程预处理中断。
- 在预处理 summary 聚合中新增 `failed_read_video` 统计，确保主 summary 能正确汇总这类单文件失败。
- 将已确认损坏的 `english/hififace/29084-cbtkoZUOR1A_83_7.mp4` 从当前 `splits/mavos_dd_english_small/test.csv` 中移除，并同步更新 `splits/mavos_dd_english_small/summary.json` 的视频数、标签数和生成器计数。
- 新增回归测试，覆盖“坏视频应记失败并继续”的场景。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mouth_roi.py'
python - <<'PY'
import csv
from pathlib import Path
needle = "english/hififace/29084-cbtkoZUOR1A_83_7.mp4"
rows = list(csv.DictReader(Path("splits/mavos_dd_english_small/test.csv").open()))
print({"test_rows": len(rows), "contains_bad_row": any(row["relative_path"] == needle for row in rows)})
PY
```

结果：通过；`test_mouth_roi.py` 4 条用例通过，当前 `test.csv` 共 `1590` 条，坏样本已不在 split 中。

## Git
- branch: `main`
- commit: `git commit -m "fix: skip unreadable mavos preprocess videos"`
