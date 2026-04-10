# Notes

## 需求
安装当前仓库 Python 环境，复核并修正根目录 requirements.txt，按单包方式完成依赖安装。

## 修改文件
- requirements.txt
- src/utils/avhubert_env.py
- tests/test_avhubert_env.py
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 将根目录 requirements 收紧到当前仓库可安装且可导入的组合：`numpy<1.24`、`opencv-python-headless==4.5.4.60`、`dlib==19.24.6`，并恢复 `torch` 条目。
- 在现有 conda 环境 `/root/shared-nvme/conda/envs/avhubert` 中使用 Python 3.8.20 和清华镜像逐包安装依赖，并补装系统级 `7z`。
- 修正 `src/utils/avhubert_env.py` 的 AV-HuBERT bootstrap 逻辑，避免 README 默认无额外参数场景下触发上游调试导入分支，导致模块导入失败或重复注册。
- 新增最小 `unittest` 回归用例，覆盖 `import_avhubert_modules()` 的默认启动场景。

## 验证
```bash
/root/shared-nvme/conda/envs/avhubert/bin/python --version
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /root/shared-nvme/conda/envs/avhubert/bin/python - <<'PY'
import torch, cv2, dlib
from src.utils.avhubert_env import import_avhubert_modules
import_avhubert_modules('third_party/av_hubert')
print(torch.__version__, cv2.__version__, dlib.__version__)
PY
which 7z && 7z | head -n 2
```

结果：通过；Python 为 `3.8.20`，`unittest` 1 条用例通过，`AV-HuBERT/fairseq` 可导入，`7z` 可用。

## Git
- branch: `main`
- commit: 待确认
