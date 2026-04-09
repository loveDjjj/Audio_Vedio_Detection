# AGENTS

## 环境

- conda 环境名：待确认
- 当前仓库可见依赖：`torch`、`dlib`、`huggingface_hub`、`PyYAML`、`opencv-python`、`7-Zip`、`third_party/av_hubert`

## 修改前先读

- 先读 [README.md](README.md)
- 先读 [docs/notes.md](docs/notes.md)
- 再读本次需求涉及的代码文件、配置文件和输出目录

## 禁止乱改

- 不要在未被要求时改算法逻辑、训练流程、配置参数值或目录结构
- 不要随意改 `splits/`、`artifacts/`、`outputs/`、`resources/` 的路径约定
- 不要直接改 `third_party/` 下的第三方代码，除非需求明确要求
- 信息不完整时写“待确认”，不要编造

## 修改规范

- 改代码时只加必要注释，注释应解释原因或关键约束，不写无信息量注释
- 改 YAML 时，新增项或重要修改要加注释
- 只在需求范围内改动，优先复用现有脚本和配置
- 输出路径、命令和文件名必须与仓库当前真实内容一致

## 文档更新规则

- 每次实际修改后都要更新 [docs/notes.md](docs/notes.md)
- 每次实际修改后都要追加更新当月日志，例如 [docs/logs/2026-04.md](docs/logs/2026-04.md)
- README 只写项目说明，不写阶段目标、长期待办或过程性记录

## Git 规范

- 分支命名建议：`docs/...`、`feat/...`、`fix/...`
- commit message 建议：`docs: ...`、`feat: ...`、`fix: ...`

## 修改后输出格式

- 列出实际修改文件
- 用一句话说明每个关键文件改了什么
- 给出验证命令和结果
- 给出风险、限制或待确认项
