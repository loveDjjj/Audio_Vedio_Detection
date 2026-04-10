# Notes

## 需求
将 mouth ROI 预处理改成严格只走 dlib CUDA CNN detector，不再回退到 CPU HOG detector。

## 修改文件
- README.md
- scripts/preprocess_av1m_mouth_roi.py
- src/preprocess/mouth_roi.py
- tests/test_mouth_roi.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 预处理入口改成严格依赖 `resources/dlib/mmod_human_face_detector.dat`；缺失时直接报错。
- 检测路径改成只调用 dlib CUDA CNN detector，不再先跑 `get_frontal_face_detector()`，也不再在 CNN 失败时回退到 CPU HOG。
- 新增 dlib CUDA/CNN 检测器构造函数，显式检查 `dlib.DLIB_USE_CUDA` 和可见 CUDA 设备数。
- 新增 `unittest` 回归用例，覆盖 CNN detector 构造约束和“CPU detector 不应被调用”的行为。

## 验证
```bash
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_mouth_roi.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
import dlib
print(dlib.DLIB_USE_CUDA, dlib.cuda.get_num_devices())
PY
```

结果：通过；`test_mouth_roi.py` 3 条用例通过，全部 `unittest` 共 12 条通过，当前环境中 `dlib.DLIB_USE_CUDA = True` 且 `dlib.cuda.get_num_devices() = 1`。

## Git
- branch: `main`
- commit: `git commit -m "fix: require cnn gpu mouth roi preprocessing"`
