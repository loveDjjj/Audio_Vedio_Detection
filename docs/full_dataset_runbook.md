# 全量数据集运行操作手册

本文档用于在服务器上按顺序运行当前仓库里的三个数据集分支。  
环境约定：

- `avhubert`：只用于 `dlib` 相关的 mouth ROI 预处理
- `oneday`：用于 split 构建、数据下载、音频缓存、训练、绘图

数据与权重路径约定：

- AV-Deepfake1M：`/data/OneDay/AV-Deepfake1M`
- FakeAVCeleb：`/data/OneDay/FakeAVCeleb`
- MAVOS-DD：`/data/OneDay/MAVOS-DD`
- AV-HuBERT 权重：`/data/OneDay/model/self_large_vox_433h.pt`

## 1. 公共准备

先进入仓库目录：

```bash
cd ~/OneDay/Audio_Vedio_Detection
```

如需下载 AV-HuBERT 权重：

```bash
conda activate oneday
python model/download_avhubert.py
```

如需检查 dlib 预处理资源是否齐全：

```bash
ls resources/dlib/shape_predictor_68_face_landmarks.dat
ls resources/dlib/mmod_human_face_detector.dat
ls resources/avhubert/20words_mean_face.npy
ls /data/OneDay/model/self_large_vox_433h.pt
```

## 2. AV-Deepfake1M

### 2.1 当前代码边界

当前仓库对 AV-Deepfake1M 的现成入口是：

- 只跑 `/data/OneDay/AV-Deepfake1M/val`
- 只使用 `real.mp4` 和 `fake_video_fake_audio.mp4`

也就是说，这里是当前仓库支持的 `real/real vs fake/fake` 运行链路，但不是 AV-Deepfake1M 全库 train+val+test 的完整协议。

### 2.2 命令顺序

如需补下载 `val`：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python dataset/download_av1m_meta.py
```

构建 split：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python dataset/build_av1m_val_real_fullfake_splits.py \
  --root /data/OneDay/AV-Deepfake1M \
  --output-dir splits/av1m_val_real_fullfake \
  --seed 42
```

做 mouth ROI 预处理：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/preprocess_av1m_mouth_roi.py
```

缓存音频特征：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/cache_av1m_audio_features.py
```

启动训练和测试：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/train_avhubert_classifier.py --config configs/avhubert_classifier.yaml
```

绘制训练曲线：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/plot_training_summary.py \
  --summary outputs/avhubert/av1m_val_real_fullfake/<timestamp>/summary.json
```

### 2.3 输出位置

- split：`splits/av1m_val_real_fullfake/`
- preprocess / audio cache：`artifacts/avhubert/av1m_val_real_fullfake/`
- 训练输出：`outputs/avhubert/av1m_val_real_fullfake/<timestamp>/`

## 3. FakeAVCeleb

### 3.1 当前代码边界

当前仓库对 FakeAVCeleb 的现成入口是：

- 只使用 `RealVideo-RealAudio`
- 只使用 `FakeVideo-FakeAudio`
- split builder 会保留全部 real，并按生成方法从 fake 中做 `1:1` 平衡采样

也就是说，这里是当前仓库支持的 `real/real vs fake/fake` 运行链路，但不是把全部 fake 样本无平衡地全部纳入训练。

### 3.2 命令顺序

构建 split：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python dataset/build_fakeavceleb_real_fullfake_splits.py \
  --root /data/OneDay/FakeAVCeleb \
  --output-dir splits/fakeavceleb_real_fullfake \
  --seed 42
```

做 mouth ROI 预处理：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/preprocess_fakeavceleb.py
```

缓存音频特征：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/cache_fakeavceleb_audio_features.py
```

启动训练和测试：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/train_fakeavceleb.py
```

绘制训练曲线：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/plot_training_summary.py \
  --summary outputs/avhubert/fakeavceleb_real_fullfake/<timestamp>/summary.json
```

### 3.3 输出位置

- split：`splits/fakeavceleb_real_fullfake/`
- preprocess / audio cache：`artifacts/avhubert/fakeavceleb_real_fullfake/`
- 训练输出：`outputs/avhubert/fakeavceleb_real_fullfake/<timestamp>/`

## 4. MAVOS-DD

### 4.1 当前代码边界

当前仓库对 MAVOS-DD 的现成入口不是整库全协议，而是：

- 只跑 `english`
- `train` 来自英语训练集
- `val` 来自英语验证集
- `test` 只取 `open_set_model=true`

如果要在当前协议下尽量跑满，就把 `train_ratio` 和 `test_ratio` 设为 `1.0`。  
这代表“跑满当前 English small 协议”，不是完整的 MAVOS-DD 全语言全协议 benchmark。

### 4.2 命令顺序

先查看 metadata 分布：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/inspect_mavos_dd_metadata.py --metadata-root /data/OneDay/MAVOS-DD
```

构建当前协议下的全量 split：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python dataset/build_mavos_dd_english_splits.py \
  --metadata-root /data/OneDay/MAVOS-DD \
  --output-dir splits/mavos_dd_english_small \
  --train-ratio 1.0 \
  --test-ratio 1.0 \
  --seed 42
```

如需补下载 split 引用到的视频文件：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python dataset/download_mavos_dd_selected_files.py \
  --split-dir splits/mavos_dd_english_small \
  --output-root /data/OneDay/MAVOS-DD
```

做 mouth ROI 预处理：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate avhubert
python scripts/preprocess_mavos_dd_english_small.py
```

缓存音频特征：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/cache_mavos_dd_english_small_audio_features.py
```

启动训练和测试：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/train_mavos_dd_english_small.py
```

绘制训练曲线：

```bash
cd ~/OneDay/Audio_Vedio_Detection
conda activate oneday
python scripts/plot_mavos_dd_english_small.py \
  --summary outputs/avhubert/mavos_dd_english_small/<timestamp>/summary.json
```

### 4.3 输出位置

- split：`splits/mavos_dd_english_small/`
- preprocess / audio cache：`artifacts/avhubert/mavos_dd_english_small/`
- 训练输出：`outputs/avhubert/mavos_dd_english_small/<timestamp>/`

## 5. 输出文件说明

训练完成后，`outputs/.../<timestamp>/` 目录里实际会写出：

- `config.yaml`
- `train.log`
- `best_head.pt`
- `last_head.pt`
- `summary.json`
- `training_curves.png`

注意：

- 磁盘上会保留训练后的模型文件 `best_head.pt` 和 `last_head.pt`
- 但当前 `.gitignore` 只放开 `outputs/` 下的 `yaml/json/png`
- 所以模型权重和日志默认不会被 git 跟踪

## 6. 当前需要你额外确认的点

- 如果你说的“全量”是指三个数据集都跑完整官方协议，那么当前仓库还不完全支持
- 当前仓库可直接运行的是：
  - AV1M：`val` 子集里的 `real.mp4` vs `fake_video_fake_audio.mp4`
  - FakeAVCeleb：`RealVideo-RealAudio` vs `FakeVideo-FakeAudio` 的平衡版本
  - MAVOS-DD：English-only 协议，且 test 只取 `open_set_model=true`
- 如果下一步要把这三条链都扩成真正的全量协议，需要继续补 split builder / 配置 / 文档
