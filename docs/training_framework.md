# 训练与预处理框架说明

本文档说明当前仓库中三套数据集共用的训练框架和预处理框架，包括：

- 整体执行链路
- 每一步依赖的环境与资源
- 每一步的输入、输出和落盘位置
- 哪些阶段主要跑在 CPU，哪些主要跑在 GPU
- 当前默认配置大小
- `configs/*.yaml` 这 9 份配置文件各自负责什么

当前覆盖的数据集协议：

- `AV-Deepfake1M`：官方 `train + val` 中的 `real.mp4` vs `fake_video_fake_audio.mp4`
- `FakeAVCeleb`：全量 `RealVideo-RealAudio` vs `FakeVideo-FakeAudio`
- `MAVOS-DD`：官方 `train / validation / test` 中的 `real/real` vs `fake/fake`

## 1. 总体链路

三套数据集共用同一条主链路，只是 split builder 和路径配置不同。

```text
原始数据/metadata
  -> split builder 生成 train.csv / val.csv / test.csv
  -> manifest builder 生成 train.list / val.list / test.list / all.list
  -> detect 阶段生成 landmarks(.pkl)
  -> align 阶段生成 mouth ROI(.mp4)
  -> audio cache 阶段生成 stacked logfbank(.npy)
  -> dataset / collate 组 batch
  -> frozen AV-HuBERT backbone + Linear probe
  -> train / val / test
  -> outputs/.../<timestamp>/ 写出模型、日志、summary 和曲线图
```

## 2. 环境与依赖

### 2.1 代码级依赖

当前框架直接依赖以下组件：

- `torch`
- `dlib`
- `opencv-python`
- `python_speech_features`
- `numpy`
- `PyYAML`
- `ffmpeg`
- `third_party/av_hubert`
- `resources/dlib/mmod_human_face_detector.dat`
- `resources/dlib/shape_predictor_68_face_landmarks.dat`
- `resources/avhubert/20words_mean_face.npy`
- `/data/OneDay/model/self_large_vox_433h.pt`

### 2.2 当前服务器环境约定

按当前服务器实测，建议这样使用：

- `oneday`
  - 用于 `detect` / `align` 预处理
  - 当前服务器上也已经实测可以运行 audio cache，因为这里有 `dlib` 且有 `python_speech_features`
- `avhubert`
  - 用于训练、测试、绘图，以及依赖 AV-HuBERT / fairseq 的主干加载
  - 如果这个环境里没有 `python_speech_features`，不要在这里跑 audio cache

### 2.3 当前默认硬件

当前配置默认按以下机器写死：

- `2 x RTX 4090 48G`
- `72 CPU cores`

对应默认并发：

- `detect`：`devices=[0,1]`，`workers_per_device=3`，`cpu_threads_per_worker=8`
- `align`：`devices=[]`，`workers_per_device=6`，`cpu_threads_per_worker=8`
- `audio cache`：`num_procs=12`，`cpu_threads_per_worker=4`
- `train`：`devices=[0,1]`，`batch_size=4`，`num_workers=8`

## 3. 每一步的输入、输出、算力归属与入口

| 步骤 | 入口脚本 | 主要输入 | 主要输出 | 主要算力 | 推荐环境 |
| --- | --- | --- | --- | --- | --- |
| 1. 构建 split | [dataset/build_av1m_official_real_fullfake_splits.py](</O:/AI Code/Audio_Vedio_Detection/dataset/build_av1m_official_real_fullfake_splits.py>) / [dataset/build_fakeavceleb_real_fullfake_splits.py](</O:/AI Code/Audio_Vedio_Detection/dataset/build_fakeavceleb_real_fullfake_splits.py>) / [dataset/build_mavos_dd_real_fullfake_splits.py](</O:/AI Code/Audio_Vedio_Detection/dataset/build_mavos_dd_real_fullfake_splits.py>) | metadata、原始视频存在性 | `splits/.../*.csv`、`summary.json` | CPU | `avhubert` |
| 2. 生成 manifest | [scripts/build_avhubert_manifests.py](</O:/AI Code/Audio_Vedio_Detection/scripts/build_avhubert_manifests.py>) 或由预处理自动触发 | `train.csv`/`val.csv`/`test.csv` | `train.list`/`val.list`/`test.list`/`all.list` | CPU | `avhubert` |
| 3. detect | [scripts/preprocess_av1m_mouth_roi.py](</O:/AI Code/Audio_Vedio_Detection/scripts/preprocess_av1m_mouth_roi.py>) 等 | 原始 `.mp4`、`all.list`、dlib 资源 | `landmarks/*.pkl` | GPU + CPU | `oneday` |
| 4. align | 同上，切换 `--config ..._preprocess_align.yaml` | 原始 `.mp4`、`landmarks/*.pkl`、均值脸模板 | `mouth_roi/*.mp4` | CPU | `oneday` |
| 5. audio cache | [scripts/cache_av1m_audio_features.py](</O:/AI Code/Audio_Vedio_Detection/scripts/cache_av1m_audio_features.py>) 等 | 原始 `.mp4`、ffmpeg | `audio_features/*.npy` | CPU | 当前服务器实测 `oneday` |
| 6. 训练/验证/测试 | [scripts/train_avhubert_classifier.py](</O:/AI Code/Audio_Vedio_Detection/scripts/train_avhubert_classifier.py>) / thin wrapper | split CSV、mouth ROI、audio features、AV-HuBERT checkpoint | `best_head.pt`、`last_head.pt`、`summary.json` 等 | GPU | `avhubert` |
| 7. 绘图 | [scripts/plot_training_summary.py](</O:/AI Code/Audio_Vedio_Detection/scripts/plot_training_summary.py>) / [scripts/plot_mavos_dd_real_fullfake.py](</O:/AI Code/Audio_Vedio_Detection/scripts/plot_mavos_dd_real_fullfake.py>) | `summary.json` | `training_curves.png` | CPU | `avhubert` |

## 4. Split 构建阶段

### 4.1 作用

这一阶段负责把原始 metadata 变成统一的 `train.csv` / `val.csv` / `test.csv`。  
后续所有阶段都只认这些 CSV，不再直接扫描原始 metadata。

### 4.2 输入

- AV1M：`train_metadata.json`、`val_metadata.json`
- FakeAVCeleb：`meta_data.csv`
- MAVOS-DD：本地 Arrow metadata

### 4.3 输出

每套协议都会生成：

- `splits/<protocol>/train.csv`
- `splits/<protocol>/val.csv`
- `splits/<protocol>/test.csv`
- `splits/<protocol>/summary.json`

### 4.4 CSV 的角色

训练链路最关键的字段是：

- `relative_path`
- `label`

其中：

- `relative_path` 指向原始视频的相对路径
- `label=0` 表示 `real`
- `label=1` 表示 `fake`

AV1M 和 MAVOS-DD 还会额外写一些便于排查的数据集字段，例如 `clip_key`、`person_id`、`generative_method` 等。

## 5. Manifest 生成阶段

### 5.1 作用

预处理阶段不会直接读取 CSV，而是先把 CSV 中的 `relative_path` 转成 manifest：

- `train.list`
- `val.list`
- `test.list`
- `all.list`

### 5.2 输入

- `splits/.../*.csv`

### 5.3 输出

- `artifacts/.../manifests/train.list`
- `artifacts/.../manifests/val.list`
- `artifacts/.../manifests/test.list`
- `artifacts/.../manifests/all.list`
- `artifacts/.../manifests/summary.json`

### 5.4 文件格式

manifest 中每一行是一个不带 `.mp4` 后缀的相对路径 ID。  
例如 CSV 里如果是 `train/person_x/clip_y/real.mp4`，manifest 里会写成 `train/person_x/clip_y/real`。

### 5.5 是否需要手动运行

通常不需要。  
[src/preprocess/runtime.py](</O:/AI Code/Audio_Vedio_Detection/src/preprocess/runtime.py:1>) 在执行预处理前会自动调用 [src/preprocess/manifest_builder.py](</O:/AI Code/Audio_Vedio_Detection/src/preprocess/manifest_builder.py:1>) 先生成 manifest。

## 6. Detect 阶段

### 6.1 作用

detect 阶段的目标是：

- 读取原始视频
- 对每一帧做人脸检测
- 对检测到的人脸做 68 点关键点回归
- 把逐帧 landmarks 缓存成 `.pkl`

### 6.2 主要输入

- 原始视频：`raw_video_root / <relative_path>.mp4`
- `all.list`
- `resources/dlib/mmod_human_face_detector.dat`
- `resources/dlib/shape_predictor_68_face_landmarks.dat`

### 6.3 主要输出

- `landmarks/<relative_path>.pkl`

### 6.4 CPU / GPU 归属

detect 不是纯 GPU 流程，而是混合流程：

- GPU
  - dlib CNN face detector
- CPU
  - `cv2.VideoCapture` 解码视频
  - dlib `shape_predictor` 68 点回归
  - landmarks 序列写入 `.pkl`
  - 多进程调度和日志落盘

### 6.5 当前实现细节

当前 [src/preprocess/mouth_roi.py](</O:/AI Code/Audio_Vedio_Detection/src/preprocess/mouth_roi.py:1>) 已做了两项关键优化：

- dlib CNN 检测支持按帧批量送 GPU，参数来自 `preprocess.detector_batch_size`
- `stage=all` 时不再重复读同一视频两次

但当前默认入口已经不再推荐 `stage=all`，而是显式拆成 `detect` 和 `align` 两步。

## 7. Align 阶段

### 7.1 作用

align 阶段的目标是：

- 读取原始视频
- 读取 detect 阶段保存的 landmarks
- 对 landmarks 做插值与平滑
- 按均值脸模板做几何对齐
- 裁出 mouth ROI
- 重新编码为 `.mp4`

### 7.2 主要输入

- 原始视频：`raw_video_root / <relative_path>.mp4`
- landmarks：`landmark_root / <relative_path>.pkl`
- `resources/avhubert/20words_mean_face.npy`

### 7.3 主要输出

- `mouth_roi/<relative_path>.mp4`

### 7.4 CPU / GPU 归属

align 当前是 CPU-only：

- CPU
  - 原始视频解码
  - landmarks 插值
  - `skimage.transform` 对齐
  - 嘴部裁剪
  - `cv2.VideoWriter` 重编码输出
- GPU
  - 不参与

### 7.5 为什么能 CPU-only

现在 [src/preprocess/runtime.py](</O:/AI Code/Audio_Vedio_Detection/src/preprocess/runtime.py:1>) 已经按 stage 区分 CUDA 需求：

- `detect` / `all`：要求 CUDA
- `align`：不要求 CUDA，`runtime.devices` 可以为空

因此现在的 `*_preprocess_align.yaml` 都是明确的 CPU-only 配置。

## 8. Audio Cache 阶段

### 8.1 作用

这一阶段把原始 mp4 中的音轨提前抽出来，转成 AV-HuBERT 训练时直接读取的 `.npy` 特征。

### 8.2 主要输入

- 原始视频：`raw_video_root / <relative_path>.mp4`
- `ffmpeg`

### 8.3 主要输出

- `audio_features/<relative_path>.npy`

### 8.4 特征流程

[src/data/audio_features.py](</O:/AI Code/Audio_Vedio_Detection/src/data/audio_features.py:1>) 当前做的是：

1. `ffmpeg` 把视频音轨转成 `16kHz`、单声道、`PCM s16le`
2. `python_speech_features.logfbank` 提取 26 维 logfbank
3. `stack_order_audio=4` 时，把连续 4 帧堆叠成 104 维
4. 保存成 `float32` 的 `.npy`

### 8.5 CPU / GPU 归属

这一阶段完全在 CPU 上：

- `ffmpeg` 子进程
- `numpy`
- `python_speech_features`
- 文件写盘

## 9. Dataset / Collate 阶段

### 9.1 作用

训练脚本不会直接读原始视频，而是读取：

- mouth ROI 视频
- `.npy` 音频特征
- split CSV 标签

### 9.2 主要实现

- 数据集类：[src/data/av1m_mouth_roi_dataset.py](</O:/AI Code/Audio_Vedio_Detection/src/data/av1m_mouth_roi_dataset.py:1>)
- batch 拼装：[src/data/collate.py](</O:/AI Code/Audio_Vedio_Detection/src/data/collate.py:1>)

### 9.3 输入

- `split_dir/*.csv`
- `mouth_roi_root/*.mp4`
- `audio_feature_root/*.npy`
- `third_party/av_hubert`

### 9.4 输出

collate 后的 batch 主要包含：

- `audio`: `(B, 104, T)`
- `video`: `(B, 1, T, 88, 88)`
- `padding_mask`: `(B, T)`
- `labels`: `(B,)`
- `relative_paths`

### 9.5 CPU / GPU 归属

- CPU
  - DataLoader worker 读取 `.mp4` / `.npy`
  - mouth ROI 图像变换
  - padding / 截断 / batch 组装
- GPU
  - batch 搬运到 GPU 后才开始参与

## 10. 模型与训练阶段

### 10.1 模型结构

当前模型固定是：

- 冻结的 AV-HuBERT backbone
- 单层线性分类头 `Linear(1024, 1)`

实现见：

- [src/models/avhubert_backbone.py](</O:/AI Code/Audio_Vedio_Detection/src/models/avhubert_backbone.py:1>)
- [src/models/binary_detector.py](</O:/AI Code/Audio_Vedio_Detection/src/models/binary_detector.py:1>)

### 10.2 前向流程

1. backbone 从音频和视频提取时序特征
2. 线性头得到逐帧 `frame_logits`
3. 用 `logsumexp` 聚合成视频级 `video_logits`
4. 用 `BCEWithLogitsLoss` 训练二分类

### 10.3 训练 / 验证 / 测试

主入口是：

- [scripts/train_avhubert_classifier.py](</O:/AI Code/Audio_Vedio_Detection/scripts/train_avhubert_classifier.py:1>)

薄封装入口是：

- [scripts/train_fakeavceleb.py](</O:/AI Code/Audio_Vedio_Detection/scripts/train_fakeavceleb.py:1>)
- [scripts/train_mavos_dd_real_fullfake.py](</O:/AI Code/Audio_Vedio_Detection/scripts/train_mavos_dd_real_fullfake.py:1>)

训练过程：

1. 解析 YAML
2. 解析 `train.devices`
3. 单卡直接训练，双卡以上自动进入 DDP
4. 构建 `train_loader` / `val_loader` / `test_loader`
5. epoch 内调用 [src/train/engine.py](</O:/AI Code/Audio_Vedio_Detection/src/train/engine.py:1>) 的 `run_epoch()`
6. 每轮保存 `last_head.pt`
7. 用验证集 `f1` 选出 `best_head.pt`
8. 训练结束后加载 `best_head.pt` 跑测试集
9. 生成 `summary.json` 和曲线图

### 10.4 CPU / GPU 归属

- GPU
  - AV-HuBERT backbone 前向
  - 线性头前向与反向
  - loss 计算
- CPU
  - DataLoader 取数
  - 指标汇总
  - `summary.json`、日志、曲线图写盘

### 10.5 当前训练入口的限制

当前 [src/train/runtime.py](</O:/AI Code/Audio_Vedio_Detection/src/train/runtime.py:1>) 和 [scripts/train_avhubert_classifier.py](</O:/AI Code/Audio_Vedio_Detection/scripts/train_avhubert_classifier.py:1>) 都按 GPU-only 写：

- `train.devices` 必须是 CUDA 设备
- 不支持 CPU-only 训练

## 11. 输出产物

每次训练会在：

- `outputs/avhubert/<protocol>/<timestamp>/`

下面写出：

- `config.yaml`
- `train.log`
- `train_rank*.log`（多卡时）
- `best_head.pt`
- `last_head.pt`
- `summary.json`
- `training_curves.png`

其中：

- `.pt` 是训练后的分类头权重
- `summary.json` 是最核心的实验摘要
- `training_curves.png` 是从 `summary.json` 画出的训练曲线

## 12. 九份 YAML 的角色

| 配置文件 | 作用 |
| --- | --- |
| [configs/avhubert_classifier.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/avhubert_classifier.yaml>) | AV1M 训练与 audio cache 主配置 |
| [configs/fakeavceleb_classifier.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/fakeavceleb_classifier.yaml>) | FakeAVCeleb 训练与 audio cache 主配置 |
| [configs/mavos_dd_real_fullfake_classifier.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/mavos_dd_real_fullfake_classifier.yaml>) | MAVOS-DD 训练与 audio cache 主配置 |
| [configs/avhubert_preprocess.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/avhubert_preprocess.yaml>) | AV1M detect 配置 |
| [configs/avhubert_preprocess_align.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/avhubert_preprocess_align.yaml>) | AV1M align 配置 |
| [configs/fakeavceleb_preprocess.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/fakeavceleb_preprocess.yaml>) | FakeAVCeleb detect 配置 |
| [configs/fakeavceleb_preprocess_align.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/fakeavceleb_preprocess_align.yaml>) | FakeAVCeleb align 配置 |
| [configs/mavos_dd_real_fullfake_preprocess.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/mavos_dd_real_fullfake_preprocess.yaml>) | MAVOS-DD detect 配置 |
| [configs/mavos_dd_real_fullfake_preprocess_align.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/mavos_dd_real_fullfake_preprocess_align.yaml>) | MAVOS-DD align 配置 |

## 13. 默认配置大小汇总

### 13.1 预处理默认值

共享默认值：

- `crop_width=96`
- `crop_height=96`
- `start_idx=48`
- `stop_idx=68`
- `window_margin=12`
- `fps=25`
- `detector_batch_size=32`

detect 默认值：

- `stage=detect`
- `devices=[0,1]`
- `workers_per_device=3`
- `cpu_threads_per_worker=8`

align 默认值：

- `stage=align`
- `devices=[]`
- `workers_per_device=6`
- `cpu_threads_per_worker=8`

### 13.2 训练默认值

AV1M / FakeAVCeleb 共享默认值：

- `epochs=10`
- `batch_size=4`
- `num_workers=8`
- `learning_rate=0.001`
- `weight_decay=0.0001`
- `grad_clip_norm=5.0`
- `amp=false`
- `max_frames=300`
- `image_crop_size=88`
- `image_mean=0.421`
- `image_std=0.165`
- `horizontal_flip_prob=0.5`
- `stack_order_audio=4`

MAVOS-DD 唯一明显差异：

- `epochs=100`

## 14. 推荐阅读顺序

如果你想继续看代码，建议按这个顺序读：

1. [docs/full_dataset_runbook.md](</O:/AI Code/Audio_Vedio_Detection/docs/full_dataset_runbook.md>)
2. [configs/avhubert_preprocess.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/avhubert_preprocess.yaml>)
3. [configs/avhubert_preprocess_align.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/avhubert_preprocess_align.yaml>)
4. [src/preprocess/runtime.py](</O:/AI Code/Audio_Vedio_Detection/src/preprocess/runtime.py>)
5. [src/preprocess/mouth_roi.py](</O:/AI Code/Audio_Vedio_Detection/src/preprocess/mouth_roi.py>)
6. [configs/avhubert_classifier.yaml](</O:/AI Code/Audio_Vedio_Detection/configs/avhubert_classifier.yaml>)
7. [src/data/audio_cache_runtime.py](</O:/AI Code/Audio_Vedio_Detection/src/data/audio_cache_runtime.py>)
8. [src/data/av1m_mouth_roi_dataset.py](</O:/AI Code/Audio_Vedio_Detection/src/data/av1m_mouth_roi_dataset.py>)
9. [src/data/collate.py](</O:/AI Code/Audio_Vedio_Detection/src/data/collate.py>)
10. [src/models/avhubert_backbone.py](</O:/AI Code/Audio_Vedio_Detection/src/models/avhubert_backbone.py>)
11. [src/models/binary_detector.py](</O:/AI Code/Audio_Vedio_Detection/src/models/binary_detector.py>)
12. [scripts/train_avhubert_classifier.py](</O:/AI Code/Audio_Vedio_Detection/scripts/train_avhubert_classifier.py>)
