# 初步方案

## 1. 参考工作与核心判断

本方案首先参考 **Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection**（下文简称 **SSR-DFD**）。这项工作的核心范式很清晰：先验证 **预训练自监督表征本身是否已经足够强**，再决定是否有必要引入更复杂的检测结构。

| 项目 | 内容 |
| --- | --- |
| 核心思路 | 使用预训练好的自监督模型作为 backbone |
| 参数策略 | **冻结 backbone**，仅训练轻量分类头 |
| 研究目标 | 比较不同自监督表征在 audio-visual deepfake detection 中的效果 |
| 对本方案的直接启发 | 先做一个结构尽量简单、结论尽量清晰的 baseline |

其中，文中 **AV-HuBERT (audio-visual)** 是一个很强的 baseline。SSR-DFD 论文表 2 中，AV-HuBERT 的 cross-dataset 结果如下：

| 训练集 | 测试集 | 论文表 2 数值 |
| --- | --- | --- |
| FAVC | FAVC | <font color="#d83931">100.0</font> |
| AV1M | FAVC | <font color="#d83931">99.5</font> |
| AV1M | AV1M | <font color="#d83931">99.9</font> |
| FAVC | AV1M | <font color="#d83931">94.5</font> |

> <font color="#d83931">关键信号：</font> 即使只冻结自监督模型、只训练一个非常简单的分类头，AV-HuBERT 依然能取得很强的效果。这说明其表征本身已经携带了较强的 deepfake 相关信息。因此，本方案的第一步应先以这一范式作为 baseline。

## 2. Baseline 方案

本阶段按 **“自行搭建训练框架”** 的前提推进，先尽量复现 SSR-DFD 的基本设置，避免过早引入额外变量。

| 模块 | 设计 | 说明 |
| --- | --- | --- |
| Backbone | **AV-HuBERT (audio-visual)** | 加载官方预训练权重，**冻结 backbone** |
| 分类头 | 线性层 + `log-sum-exp pooling` | 保持与 SSR-DFD 风格一致 |
| 监督信号 | Video-level fake/real label | 先完成视频级真假分类 |
| 目标 | 建立强而简单的 baseline | 便于后续观察“表征够不够用” |

## 3. 数据集与实验安排

### 3.1 数据集资源总表

| Level | 数据集 | 作用定位 | 关键特点 | 资源地址 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Level-1 | **FakeAVCeleb (FAVC)** | 第一阶段 baseline 验证 | 公开仓库给出 **20,000** 个视频，其中 **500 real / 19,500 fake** | [GitHub](https://github.com/DASH-Lab/FakeAVCeleb) | 下载通常需填写申请表 |
| Level-1 | **AV-Deepfake1M (AV1M)** | 第一阶段 baseline 验证 + 跨数据集测试 | **100 万+** 视频，覆盖 **2000+** 人物，包含 audio / video / audio-visual manipulation | [GitHub](https://github.com/ControlNet/AV-Deepfake1M) | 适合做更大规模的泛化观察 |
| Level-2 | **MAVOS-DD** | 第二阶段开放集泛化测试 | **250+ 小时**、**8 种语言**、约 **60%** 为生成内容，强调 open-set 评测 | [Hugging Face](https://huggingface.co/datasets/unibuc-cs/MAVOS-DD) / [GitHub](https://github.com/CroitoruAlin/MAVOS-DD) | 更适合检验未见语言与未见生成器泛化 |
| Level-2 | **MVAD** | 第二阶段多生成器扩展测试 | 覆盖 **20+** 生成方法，包含 **F-F / R-F / F-R / R-R** 四类模态组合 | [GitHub](https://github.com/HuMengXue0104/MVAD) / [Hugging Face](https://huggingface.co/datasets/mengxuebobo/MVAD) | 更适合观察多场景、多模态组合下的鲁棒性 |

### 3.2 整体推进路径

| 阶段 | 目标 | 预期产出 |
| --- | --- | --- |
| 第一步 | 参考 SSR-DFD，建立 **AV-HuBERT + 简单分类头** baseline | 在 **FAVC / AV1M** 上得到基础结果，并完成 in-domain / cross-dataset 对齐 |
| 第二步 | 保持相同框架，迁移到 **MAVOS-DD / MVAD** | 观察在更多生成器、更复杂分布、开放集条件下的泛化表现 |

整体上，这个方案的逻辑非常明确：  
先用 **简单而强的 baseline** 建立参照，再逐步把问题推向 **更多生成器、更复杂分布、更强泛化压力** 的场景。

