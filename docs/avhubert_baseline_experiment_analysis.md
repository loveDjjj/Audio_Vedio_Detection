# AV-HuBERT Baseline Pilot Analysis

> [!abstract]
> 本文档基于 [[Initial Plan]]，整理当前已经完成的两类实验：
> 1. `AV-Deepfake1M (AV1M)` 的 in-domain pilot baseline
> 2. `MAVOS-DD` 英语小样本的 `open_set_model` pilot baseline
>
> 目标是回答两个问题：
> - 冻结 `AV-HuBERT` + 轻量线性头，是否已经足够构成一个强 baseline？
> - 同一套框架迁移到更难、更开放的分布后，性能会如何变化？

## 1. 与初步方案的对应关系

[[Initial Plan]] 中的核心判断有两条：

1. 先做一个尽量简单的 SSR-DFD 风格 baseline：
   - `frozen AV-HuBERT backbone`
   - `Linear(1024, 1)` 分类头
   - `log-sum-exp pooling`
2. 先在相对直接的设置上建立参照，再迁移到更复杂、更开放的分布上观察泛化。

当前已经完成的工作与该计划的对应关系如下：

| 计划项 | 当前状态 | 备注 |
| --- | --- | --- |
| AV-HuBERT + 简单分类头 baseline | 已完成 | 当前训练头为 frozen backbone + 单层线性 probe |
| AV1M baseline 验证 | 已完成 pilot | 使用 `AV-Deepfake1M/val` 子集中的 `real.mp4` 和 `fake_video_fake_audio.mp4` |
| MAVOS-DD 更复杂分布测试 | 已完成 pilot | 当前仅做英语小样本 + `open_set_model=true` 的 unseen-generator 测试 |
| FAVC / 真正 cross-dataset 对齐 | 未完成 | 还未跑 FAVC，也未完成官方意义上的 cross-dataset protocol |
| MVAD 扩展测试 | 未完成 | 目前尚未接入 |

> [!info]
> 因此，当前最准确的表述不是“完整完成了初步方案”，而是：
> **已经完成了第一阶段 baseline 的核心搭建与两组 pilot 实验，并拿到了可解释的初步结果。**

## 2. 统一实验设定

所有当前实验都复用了同一个核心检测框架：

| 模块 | 当前实现 |
| --- | --- |
| Backbone | 预训练 `AV-HuBERT (audio-visual)` |
| 参数策略 | backbone 全冻结，仅训练分类头 |
| 分类头 | `Linear(1024, 1)` |
| 聚合方式 | frame-level logits 经 `logsumexp` 做 video-level pooling |
| 输入 | `mouth ROI video + cached audio logfbank features` |
| 损失 | `BCEWithLogitsLoss` |
| 训练设备 | 8 卡 DDP (`devices=[0,1,2,3,4,5,6,7]`) |

统一不变的关键训练参数：

| 参数 | 数值 |
| --- | --- |
| `max_frames` | `300` |
| `image_crop_size` | `88` |
| `learning_rate` | `1e-3` |
| `weight_decay` | `1e-4` |
| `grad_clip_norm` | `5.0` |
| `amp` | `false` |

## 3. 数据集与当前实际使用的数据切片

### 3.1 AV-Deepfake1M (AV1M)

当前 AV1M 实验并没有覆盖整个 `AV-Deepfake1M`，而是使用了仓库内已经准备好的 `val` 子集，并进一步只保留：

- `real.mp4`
- `fake_video_fake_audio.mp4`

对应训练输出目录：
- [[outputs/avhubert/av1m_val_real_fullfake/20260411-040113]]

实际进入训练链的样本规模：

| Split | 可用样本数 |
| --- | ---: |
| Train | `22996` |
| Val | `2874` |
| Test | `2874` |

补充说明：
- 该结果对应的是 **in-domain pilot**，不是严格 cross-dataset 评测。
- 当前 `AV1M` split 仍带有“同一原始来源分布内随机切分”的性质，因此结果应理解为“当前仓库协议下的高性能 baseline”，而不是论文级严格泛化结论。

### 3.2 MAVOS-DD English Small

出于数据规模和试验成本考虑，当前对 `MAVOS-DD` 只做了英语小样本 pilot：

- 训练集：英语 `train` split 按生成器分层抽样 `1/5`
- 验证集：英语 `validation` 全量
- 测试集：英语 `test` 中 `open_set_model=true` 按生成器分层抽样 `1/5`

这意味着当前 MAVOS-DD 实验重点不是“多语言泛化”，而是：

> **英语条件下，对未见生成器 / unseen model 的初步泛化能力。**

对应训练输出目录：
- [[outputs/avhubert/mavos_dd_english_small/20260412-051856]]
- [[outputs/avhubert/mavos_dd_english_small/20260412-052419]]
- [[outputs/avhubert/mavos_dd_english_small/20260412-054048]]

当前 split 规模与有效样本规模如下：

| Split | Split 原始样本数 | 经过 mouth ROI 后可用样本数 |
| --- | ---: | ---: |
| Train | `1277` | `1237` |
| Val | `1079` | `1036` |
| Test | `1590` | `1477` |

预处理失败的主要原因：
- `no_landmarks`: `193`
- `read_video_failed`: `3`

测试集生成器组成（当前 pilot）：

| Generator | 样本数 |
| --- | ---: |
| `hififace` | `422` |
| `roop` | `291` |
| `sonic` | `67` |
| `real` | `810` |

> [!warning]
> 当前 MAVOS-DD pilot 不是“完整的 MAVOS-DD out-of-model benchmark”，而是：
> **English-only + open_set_model small pilot**。

## 4. 当前已完成的实验配置

### 4.1 AV1M baseline

来源配置：
- [[outputs/avhubert/av1m_val_real_fullfake/20260411-040113/config.yaml]]

关键参数：

| 项目 | 数值 |
| --- | --- |
| `epochs` | `10` |
| `batch_size` | `4` |
| `num_workers` | `16` |
| `devices` | `8 GPUs` |

### 4.2 MAVOS-DD English Small：Run A

来源配置：
- [[outputs/avhubert/mavos_dd_english_small/20260412-051856/config.yaml]]

关键参数：

| 项目 | 数值 |
| --- | --- |
| `epochs` | `10` |
| `batch_size` | `8` |
| `num_workers` | `16` |
| `devices` | `8 GPUs` |

### 4.3 MAVOS-DD English Small：Run B

来源配置：
- [[outputs/avhubert/mavos_dd_english_small/20260412-052419/config.yaml]]

关键参数：

| 项目 | 数值 |
| --- | --- |
| `epochs` | `50` |
| `batch_size` | `4` |
| `num_workers` | `16` |
| `devices` | `8 GPUs` |

### 4.4 MAVOS-DD English Small：Run C

来源配置：
- [[outputs/avhubert/mavos_dd_english_small/20260412-054048/config.yaml]]

关键参数：

| 项目 | 数值 |
| --- | --- |
| `epochs` | `100` |
| `batch_size` | `4` |
| `num_workers` | `16` |
| `devices` | `8 GPUs` |

## 5. 结果汇总

### 5.1 AV1M baseline 结果

来源：
- [[outputs/avhubert/av1m_val_real_fullfake/20260411-040113/summary.json]]

| 指标 | 数值 |
| --- | ---: |
| `best_epoch` | `10` |
| `best_val_f1` | `0.9871` |
| `test_accuracy` | `0.9892` |
| `test_precision` | `0.9966` |
| `test_recall` | `0.9825` |
| `test_f1` | `0.9895` |
| `test_loss` | `0.0395` |

曲线：

![[outputs/avhubert/av1m_val_real_fullfake/20260411-040113/training_curves.png]]

解读：
- 该结果非常高，说明 **冻结 AV-HuBERT + 线性头** 在当前 AV1M in-domain 设置下已经足够强。
- 训练和验证曲线收敛很快，`10 epoch` 已足够。
- 这与 [[Initial Plan]] 中“AV-HuBERT 作为强 baseline”的判断是一致的。

但这里必须加一个保留意见：

> [!warning]
> 当前 AV1M 结果更适合解释为“当前仓库协议下的强 in-domain baseline”，而不是严格论文级 cross-dataset 结论。

### 5.2 MAVOS-DD English Small 结果对比

结果来源：
- [[outputs/avhubert/mavos_dd_english_small/20260412-051856/summary.json]]
- [[outputs/avhubert/mavos_dd_english_small/20260412-052419/summary.json]]
- [[outputs/avhubert/mavos_dd_english_small/20260412-054048/summary.json]]

| Run | Epochs | Batch Size | Best Epoch | Best Val F1 | Test Acc | Test Precision | Test Recall | Test F1 | Test Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260412-051856` | `10` | `8` | `10` | `0.7072` | `0.6405` | `0.6568` | `0.9529` | `0.7776` | `0.8898` |
| `20260412-052419` | `50` | `4` | `48` | `0.7579` | `0.6655` | `0.7503` | `0.7387` | `0.7445` | `0.6484` |
| `20260412-054048` | `100` | `4` | `99` | `0.7785` | `0.6689` | `0.7780` | `0.6967` | `0.7351` | `0.6812` |

#### Run A：10 epoch / batch size 8

曲线：

![[outputs/avhubert/mavos_dd_english_small/20260412-051856/training_curves.png]]

解读：
- 训练明显不足。
- 该 run 的 test `F1` 虽然数值高于后两个 run，但它的预测行为偏“几乎都判 fake”：
  - `recall` 很高：`0.9529`
  - `tn = 18`
  - `fp = 486`
- 这更像是一个 **偏塌缩、阈值偏置明显** 的解，而不是稳定的 open-set detector。

#### Run B：50 epoch / batch size 4

曲线：

![[outputs/avhubert/mavos_dd_english_small/20260412-052419/training_curves.png]]

解读：
- 这是第一个比较稳定、可解释的 MAVOS-DD baseline。
- `val_f1` 从 `0.69` 左右持续提升到 `0.758`。
- test 的 precision / recall 比较均衡：
  - `precision = 0.7503`
  - `recall = 0.7387`
- 从 open-set pilot 的角度看，这个 run 比 Run A 更可信。

#### Run C：100 epoch / batch size 4

曲线：

![[outputs/avhubert/mavos_dd_english_small/20260412-054048/training_curves.png]]

解读：
- `best_val_f1` 提升到了当前最高：`0.7785`
- `test_accuracy` 也是三组 MAVOS run 中最高：`0.6689`
- 但是 `test_f1` 下降到了 `0.7351`

这说明：
- `50 -> 100 epoch` 的确带来了更充分的收敛
- 但模型开始变得更保守：
  - `precision` 上升
  - `recall` 下降
- 也就是说，当前结果已经进入 **阈值敏感、收益递减的后期收敛区**

> [!summary]
> 就“训练是否充分”而言，`100 epoch` 明显优于 `10 epoch`，也略优于 `50 epoch`。
> 但就“是否值得继续无限增大 epoch”而言，答案是否定的：收益已经开始明显变小。

## 6. 对初始猜想的检验

### 猜想 1：冻结 AV-HuBERT + 简单线性头可以构成一个强 baseline

**结论：成立。**

证据：
- AV1M pilot 中，`test_f1 = 0.9895`
- MAVOS-DD English small pilot 中，即使面对 unseen generators，依然能稳定达到：
  - `val_f1 ≈ 0.76 ~ 0.78`
  - `test_acc ≈ 0.66 ~ 0.67`

解释：
- 在容易的 in-domain 条件下，这个 baseline 已经非常强。
- 在更难的 open-set 条件下，它虽不再接近“饱和”，但仍然能学到明显高于随机的有效判别能力。

### 猜想 2：同一套框架可以迁移到更复杂、更开放的数据分布

**结论：成立。**

证据：
- 从 `AV1M` 迁移到 `MAVOS-DD`，模型结构没有变化，只做了数据适配。
- 预处理、音频缓存、训练、绘图链路都能复用。
- 说明工程路径是可迁移的。

### 猜想 3：开放集 / 未见生成器场景会比 AV1M in-domain 明显更难

**结论：成立，而且差异很明显。**

对比：
- AV1M pilot：`test_f1 = 0.9895`
- MAVOS-DD English small：`test_f1 ≈ 0.735 ~ 0.745`

解释：
- AV1M 当前设置更接近同分布二分类
- MAVOS-DD 当前设置是英语 + unseen model pilot
- 指标大幅下降，说明 open-set generator generalization 的难度确实更高

### 猜想 4：继续加大 epoch 会持续明显提升效果

**结论：不完全成立。**

证据：
- `10 -> 50 epoch`：收益明显
- `50 -> 100 epoch`：仍有提升，但幅度已经明显变小
- `best_epoch = 99` 说明当前还没完全“提前停住”，但 improvement 已接近平台区

解释：
- 训练轮数不是当前最主要的瓶颈了
- 后续更值得投入的方向可能是：
  - 阈值校准
  - 学习率 / 正则化策略
  - 更严格的评估 protocol
  - 更大、更完整的 MAVOS-DD 子集

## 7. 当前阶段的总体结论

> [!success]
> 当前实验已经足以支持一个阶段性结论：
> **SSR-DFD 风格的 frozen AV-HuBERT linear probe 是一个成立且有解释力的 baseline。**

但这个结论需要分两层理解：

### 在 AV1M 上
- baseline 很强
- 说明自监督 AV-HuBERT 表征本身就带有很强的 deepfake 相关信息

### 在 MAVOS-DD English small 上
- baseline 仍然可用
- 但性能显著下降
- 说明 open-set / unseen generator 条件下，仅靠 frozen representation + 线性头还不够“轻松”

这正好符合 [[Initial Plan]] 的整体逻辑：

1. 先建立一个简单而强的 baseline
2. 再把它推到更难的分布上
3. 观察性能落差，从而判断后续是否有必要引入更复杂结构

当前结果已经给出了明确答案：

> [!note]
> **需要。**
> 至少在更开放的 MAVOS-DD setting 下，后续继续做更完整数据、阈值校准、甚至更强 head / 更严格 split protocol 都是有必要的。

## 8. 目前还不能下的结论

以下结论目前仍然**不能**直接声称已经完成：

- 还不能说已经完成 `FAVC / AV1M` 的严格 cross-dataset 对齐
- 还不能说已经完成 `MAVOS-DD` 全量的 out-of-model benchmark
- 还不能说已经完成 `MVAD` 上的多生成器扩展测试
- 还不能把当前 AV1M pilot 直接当作论文级严格泛化结果

## 9. 下一步建议

### 建议 1：固定当前 baseline 作为参照

建议保留以下两个 run 作为当前阶段参照：

- AV1M:
  - [[outputs/avhubert/av1m_val_real_fullfake/20260411-040113]]
- MAVOS-DD:
  - [[outputs/avhubert/mavos_dd_english_small/20260412-054048]]

理由：
- 前者代表“当前仓库协议下的强 in-domain baseline”
- 后者代表“当前英语 open_set_model pilot 下的最佳验证表现”

### 建议 2：MAVOS-DD 不再继续单纯增大 epoch

更合理的后续方向是：
- 加 early stopping
- 做 validation-based threshold tuning
- 补充 `closed-set model` vs `open-set model` 对比
- 扩展到更完整的英语 test，甚至更多语言

### 建议 3：如果要做更干净的“多生成器效果”分析

最好进一步区分：
- `audio_fake`
- `video_fake`
- `audio_generative_method`
- `generative_method`

因为 MAVOS-DD 本身是多模态伪造数据集，当前 pilot 不是纯视觉生成器比较。

## 10. 结论一句话版

> [!quote]
> 当前实验说明：  
> **冻结 AV-HuBERT + 线性 probe 的简单 baseline 在 AV1M 上已经非常强，在 MAVOS-DD 英语 open-set model pilot 上也具备可用的泛化能力；但开放集场景明显更难，后续研究重点不应再只是增加训练 epoch，而应转向更严格的评估协议、阈值校准和更复杂分布上的扩展验证。**
