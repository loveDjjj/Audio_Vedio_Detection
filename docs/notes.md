# Notes

## 需求
补充当前仓库所需的根目录 requirements.txt。

## 修改文件
- requirements.txt
- docs/notes.md
- docs/logs/2026-04.md

## 修改内容
- 根据当前仓库真实代码、AV-HuBERT 上游 requirements 和 vendored fairseq 依赖，新增根目录 requirements.txt。
- 更新最近一次修改摘要和月度日志，记录这次环境依赖文件补充。

## 验证
```bash
Get-Content requirements.txt
```

结果：通过

## Git
- branch: `main`
- commit: `git commit -m "build: add project requirements file"`
