# 全量数据集运行手册

本文档用于在服务器上按顺序运行当前仓库支持的 3 个数据集分支。

环境约定：
- `oneday`：mouth ROI 预处理
- `avhubert`：split 构建、元数据检查、音频缓存、训练、绘图

数据与权重路径：
- AV-Deepfake1M：`/data/OneDay/AV-Deepfake1M`
- FakeAVCeleb：`/data/OneDay/FakeAVCeleb`
- MAVOS-DD：`/data/OneDay/MAVOS-DD`
- AV-HuBERT 权重：`/data/OneDay/model/self_large_vox_433h.pt`

默认硬件配置：
- GPU：`2 x RTX 4090 48G`
- CPU：`72 cores`

预处理配置说明：
- `configs/*_preprocess.yaml`：默认 `stage=detect`，用于 GPU 检测并生成 landmarks
- `configs/*_preprocess_align.yaml`：默认 `stage=align`，用于 CPU-only 对齐、裁剪和写 mouth ROI

## 1. 公共准备

进入仓库：

```bash
cd ~/OneDay/Audio_Vedio_Detection
```

检查关键资源：

```bash
conda activate avhubert
ls resources/dlib/shape_predictor_68_face_landmarks.dat
ls resources/dlib/mmod_human_face_detector.dat
ls resources/avhubert/20words_mean_face.npy
ls /data/OneDay/model/self_large_vox_433h.pt
```

如需下载 AV-HuBERT 权重：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python model/download_avhubert.py
```

## 2. AV-Deepfake1M

### 2.1 协议范围

当前入口运行的是：
- 官方 `train_metadata.json` + `val_metadata.json`
- 只保留 `real.mp4` 和 `fake_video_fake_audio.mp4`
- 官方 `train` 全量作训练集
- 官方 `val` 按 clip 随机切成 `val/test`

### 2.2 命令顺序

构建 split：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python dataset/build_av1m_official_real_fullfake_splits.py \
  --root /data/OneDay/AV-Deepfake1M \
  --output-dir splits/av1m_official_real_fullfake \
  --seed 42
```

detect 阶段，生成 landmarks：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/preprocess_av1m_mouth_roi.py
```

align 阶段，生成 mouth ROI：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/preprocess_av1m_mouth_roi.py \
  --config configs/avhubert_preprocess_align.yaml
```

缓存音频特征：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/cache_av1m_audio_features.py
```

训练与测试：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/train_avhubert_classifier.py --config configs/avhubert_classifier.yaml
```

绘制训练曲线：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/plot_training_summary.py \
  --summary outputs/avhubert/av1m_official_real_fullfake/<timestamp>/summary.json
```

### 2.3 输出位置

- split：`splits/av1m_official_real_fullfake/`
- preprocess / audio cache：`/data/OneDay/artifacts/avhubert/av1m_official_real_fullfake/`
- 训练输出：`outputs/avhubert/av1m_official_real_fullfake/<timestamp>/`

## 3. FakeAVCeleb

### 3.1 协议范围

当前入口运行的是：
- `RealVideo-RealAudio`
- `FakeVideo-FakeAudio`
- 全量样本，按 `label/method` 分层切分 `train/val/test`

### 3.2 命令顺序

构建 split：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python dataset/build_fakeavceleb_real_fullfake_splits.py \
  --root /data/OneDay/FakeAVCeleb \
  --output-dir splits/fakeavceleb_real_fullfake \
  --seed 42
```

detect 阶段，生成 landmarks：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/preprocess_fakeavceleb.py
```

align 阶段，生成 mouth ROI：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/preprocess_fakeavceleb.py \
  --config configs/fakeavceleb_preprocess_align.yaml
```

缓存音频特征：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/cache_fakeavceleb_audio_features.py
```

训练与测试：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/train_fakeavceleb.py
```

绘制训练曲线：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/plot_training_summary.py \
  --summary outputs/avhubert/fakeavceleb_real_fullfake/<timestamp>/summary.json
```

### 3.3 输出位置

- split：`splits/fakeavceleb_real_fullfake/`
- preprocess / audio cache：`/data/OneDay/artifacts/avhubert/fakeavceleb_real_fullfake/`
- 训练输出：`outputs/avhubert/fakeavceleb_real_fullfake/<timestamp>/`

## 4. MAVOS-DD

### 4.1 协议范围

当前入口运行的是：
- 官方 `train / validation / test`
- 只保留 `real/real` 和 `fake/fake`
- 不再限制 `language`
- 不再限制 `open_set_model`

### 4.2 命令顺序

检查 metadata：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/inspect_mavos_dd_metadata.py --metadata-root /data/OneDay/MAVOS-DD
```

构建 split：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python dataset/build_mavos_dd_real_fullfake_splits.py \
  --metadata-root /data/OneDay/MAVOS-DD \
  --output-dir splits/mavos_dd_real_fullfake
```

如需补下载 split 中引用的视频：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python dataset/download_mavos_dd_selected_files.py \
  --split-dir splits/mavos_dd_real_fullfake \
  --output-root /data/OneDay/MAVOS-DD
```

detect 阶段，生成 landmarks：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/preprocess_mavos_dd_real_fullfake.py
```

align 阶段，生成 mouth ROI：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/preprocess_mavos_dd_real_fullfake.py \
  --config configs/mavos_dd_real_fullfake_preprocess_align.yaml
```

缓存音频特征：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/cache_mavos_dd_real_fullfake_audio_features.py
```

训练与测试：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/train_mavos_dd_real_fullfake.py
```

绘制训练曲线：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/plot_mavos_dd_real_fullfake.py \
  --summary outputs/avhubert/mavos_dd_real_fullfake/<timestamp>/summary.json
```

### 4.3 输出位置

- split：`splits/mavos_dd_real_fullfake/`
- preprocess / audio cache：`/data/OneDay/artifacts/avhubert/mavos_dd_real_fullfake/`
- 训练输出：`outputs/avhubert/mavos_dd_real_fullfake/<timestamp>/`

## 5. 输出文件说明

训练完成后，`outputs/.../<timestamp>/` 中通常会包含：
- `config.yaml`
- `train.log`
- `best_head.pt`
- `last_head.pt`
- `summary.json`
- `training_curves.png`

注意：
- 磁盘上会保留训练后的 `.pt` 模型文件
- 当前 `.gitignore` 只放开 `outputs/` 下的 `yaml/json/png`
- 所以模型和日志默认不会被 git 跟踪
