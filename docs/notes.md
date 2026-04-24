# Notes

## 本次需求

修复 MAVOS-DD detect 预处理阶段容易因长视频/高分辨率视频导致内存峰值过高并被系统 kill 的问题。根因是 detect 阶段原先会先把整段视频读成 BGR 帧列表，再一次性转换为整段灰度帧列表，单个 worker 会同时持有完整视频的两份帧数据。

## 实际修改文件

- src/preprocess/mouth_roi.py
- configs/mavos_dd_real_fullfake_preprocess.yaml
- tests/test_fakeavceleb_wrappers.py
- docs/notes.md
- docs/logs/2026-04.md

## 关键改动

- 将 `detect_landmarks_for_video()` 改为流式读取视频：按 `detector_batch_size` 从 `cv2.VideoCapture` 读取小批次帧，检测完一批即释放，不再整段加载。
- `stage: detect` 的 `process_manifest()` 改为直接调用流式 `detect_landmarks_for_video()`，避免走 `load_video_frames()`。
- `detect_landmarks_for_frames()` 继续保留给 `stage: all` 使用，但内部也改成按批次生成灰度帧，避免额外持有整段灰度副本。
- 将 MAVOS-DD detect 默认 `detector_batch_size` 从 `128` 降到 `16`，降低 dlib CNN 的显存和临时内存峰值。
- 补充回归测试，确保 `stage: detect` 不再调用整段视频加载函数，同时保留批量 detector 调用行为。

## 验证

```bash
python -m unittest tests.test_fakeavceleb_wrappers.PreprocessOptimizationTest -v
python -m py_compile src/preprocess/mouth_roi.py tests/test_fakeavceleb_wrappers.py
```

结果：通过。

## 服务器建议

MAVOS-DD detect 可以先用以下保守配置确认稳定：

```yaml
preprocess:
  detector_batch_size: 16

runtime:
  devices: [0, 1]
  workers_per_device: 1
  cpu_threads_per_worker: 2
```

如果稳定，再逐步提高 `workers_per_device`；优先观察 `free -h` 和 `nvidia-smi`。

## 风险与限制

- `stage: all` 仍需要为后续 align 保留帧数据，因此本次主要解决独立 `stage: detect` 的内存峰值。
- `detector_batch_size` 变小会降低单 worker 的 GPU 批处理吞吐，但优先保证 MAVOS-DD 长视频稳定跑完。

## Git

- branch: main
- commit: 待确认
