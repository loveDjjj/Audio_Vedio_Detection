# Audio_Vedio_Detection

## 项目简介

这是一个小型音视频伪造检测项目，当前仓库围绕 `AV-HuBERT` 搭建了一条 `AV-Deepfake1M` `val` 子集的二分类流程。当前训练 baseline 采用 SSR-DFD 风格的 frozen AV-HuBERT linear probe：输入原始视频中的音频和 mouth ROI 视频，输出 frame-level logits，并通过 `logsumexp` 聚合成视频级 fake/real logit。

## 项目结构

```text
configs/
  avhubert_classifier.yaml
  avhubert_preprocess.yaml
dataset/
  AV-Deepfake1M/
  download_av1m_meta.py
  build_av1m_val_real_fullfake_splits.py
model/
  self_large_vox_433h.pt
scripts/
  build_avhubert_manifests.py
  preprocess_av1m_mouth_roi.py
  train_avhubert_classifier.py
splits/
  av1m_val_real_fullfake/
src/
  data/
  models/
  preprocess/
  train/
  utils/
resources/
third_party/
  av_hubert/
artifacts/
outputs/
```

## 关键文件说明

- [configs/avhubert_classifier.yaml](configs/avhubert_classifier.yaml)：当前训练主配置，集中管理训练数据路径、模型、单卡/多卡 DDP 参数和训练参数。
- [configs/avhubert_preprocess.yaml](configs/avhubert_preprocess.yaml)：当前预处理配置，集中管理 mouth ROI 预处理路径、裁剪参数、多进程和多卡运行参数。
- [dataset/download_av1m_meta.py](dataset/download_av1m_meta.py)：下载 `AV-Deepfake1M` 的 `val` 分卷和 `val_metadata.json`，并调用 `7z` 解压。
- [dataset/build_av1m_val_real_fullfake_splits.py](dataset/build_av1m_val_real_fullfake_splits.py)：从 `val_metadata.json` 中筛出 `real.mp4` 和 `fake_video_fake_audio.mp4`，随机生成 `train/val/test` 列表。
- [scripts/build_avhubert_manifests.py](scripts/build_avhubert_manifests.py)：把 split CSV 转成 AV-HuBERT 预处理用的 `*.list`。
- [scripts/preprocess_av1m_mouth_roi.py](scripts/preprocess_av1m_mouth_roi.py)：读取独立预处理 YAML，使用 dlib CUDA CNN face detector 做多进程、多卡分片 mouth ROI 预处理，并由主进程汇总进度和结果。
- [scripts/train_avhubert_classifier.py](scripts/train_avhubert_classifier.py)：读取训练 YAML，按 `train.devices` 自动切换单卡或多卡 DDP，加载 frozen `AV-HuBERT` audio-visual backbone 和单层线性 probe，执行训练、验证和测试。

## 运行方法

如果数据和权重尚未准备好，可按以下顺序执行：

```bash
python dataset/download_av1m_meta.py
python dataset/build_av1m_val_real_fullfake_splits.py
python scripts/preprocess_av1m_mouth_roi.py
python scripts/train_avhubert_classifier.py
```

运行前需要确认以下资源已经就位：

- 数据目录：`dataset/AV-Deepfake1M/val`
- 主配置：`configs/avhubert_classifier.yaml`
- 权重文件：`model/self_large_vox_433h.pt`
- 第三方仓库：`third_party/av_hubert`
- 外部工具：`ffmpeg`
- 预处理设备：CUDA 可用的 NVIDIA GPU（当前 mouth ROI 预处理要求 dlib CUDA CNN detector，不再回退到 CPU HOG detector）
- 训练设备：可用的 NVIDIA GPU（当前训练入口要求 `CUDA`，不再回退到 CPU）
- 预处理资源：`resources/dlib/shape_predictor_68_face_landmarks.dat`、`resources/dlib/mmod_human_face_detector.dat`、`resources/avhubert/20words_mean_face.npy`

环境安装命令仓库内未完全固定，具体 conda 环境名待确认。

## 配置说明

当前使用两个配置文件：

- [configs/avhubert_preprocess.yaml](configs/avhubert_preprocess.yaml)
  - `paths`：split、raw video、landmark、mouth ROI 和预处理资源路径
  - `preprocess`：manifest 选择、裁剪参数和 strict/save_landmarks
  - `runtime`：设备列表、每卡 worker 数、每 worker CPU 线程数、主进程进度条和多进程启动方式
- [configs/avhubert_classifier.yaml](configs/avhubert_classifier.yaml)
  - `paths`：训练数据、checkpoint、第三方仓库与输出路径
  - `data`：帧数上限、裁剪尺寸、图像归一化和 batch 处理方式
  - `model`：当前只保留是否冻结 backbone；分类头固定为单层线性 probe
  - `train`：DDP 设备列表、后端、master 地址端口，以及每 GPU 的 batch size / worker 数和训练参数

## 输出说明

- `splits/av1m_val_real_fullfake/`：训练、验证、测试 CSV 及 split 统计
- `artifacts/avhubert/av1m_val_real_fullfake/manifests/`：`train.list`、`val.list`、`test.list`、`all.list`
- `artifacts/avhubert/av1m_val_real_fullfake/landmarks/`：关键点检测结果
- `artifacts/avhubert/av1m_val_real_fullfake/mouth_roi/`：裁剪后的嘴部 ROI 视频
- `artifacts/avhubert/av1m_val_real_fullfake/preprocess.log`：预处理主进程日志；worker 日志会写成 `preprocess_rank*.log`
- `outputs/avhubert/av1m_val_real_fullfake/<timestamp>/`：训练输出目录，包含 `config.yaml`、`train.log`、`best_head.pt`、`last_head.pt`、`summary.json`；多卡时其它 rank 会写 `train_rank*.log`

## 阅读顺序

1. [README.md](README.md)
2. [configs/avhubert_classifier.yaml](configs/avhubert_classifier.yaml)
3. [scripts/preprocess_av1m_mouth_roi.py](scripts/preprocess_av1m_mouth_roi.py)
4. [scripts/train_avhubert_classifier.py](scripts/train_avhubert_classifier.py)
5. `src/` 下对应模块
