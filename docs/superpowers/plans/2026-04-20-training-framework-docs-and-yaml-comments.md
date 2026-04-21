# Training Framework Docs And YAML Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a detailed training/preprocess framework document and enrich every `configs/*.yaml` file with clear Chinese comments that explain purpose, dependencies, inputs/outputs, compute placement, and default values.

**Architecture:** Keep behavior unchanged. Document the existing split build, manifest, detect, align, audio cache, dataset loading, frozen AV-HuBERT, linear probe, DDP training, and output-writing pipeline as it exists today. Add comments directly in the nine YAML files so operators can understand each section without reading source code first.

**Tech Stack:** Markdown, YAML, existing Python entrypoints and config loader, AV-HuBERT, dlib, ffmpeg, PyYAML.

---

### Task 1: Capture The Current Pipeline Facts

**Files:**
- Modify: `docs/superpowers/plans/2026-04-20-training-framework-docs-and-yaml-comments.md`
- Check: `README.md`
- Check: `docs/notes.md`
- Check: `scripts/train_avhubert_classifier.py`
- Check: `src/preprocess/runtime.py`
- Check: `src/preprocess/mouth_roi.py`
- Check: `src/data/audio_cache_runtime.py`
- Check: `src/data/audio_features.py`
- Check: `src/models/avhubert_backbone.py`
- Check: `src/models/binary_detector.py`
- Check: `src/data/av1m_mouth_roi_dataset.py`
- Check: `dataset/build_av1m_official_real_fullfake_splits.py`
- Check: `dataset/build_fakeavceleb_real_fullfake_splits.py`
- Check: `dataset/build_mavos_dd_real_fullfake_splits.py`

- [ ] **Step 1: Record the concrete facts to preserve**

Capture these facts and use them as the source of truth:

- Preprocess default configs now run `stage: detect`
- Dedicated align configs now run `stage: align`
- `detect` uses dlib CNN face detection with CUDA plus CPU landmark prediction and CPU video I/O
- `align` uses saved landmarks plus CPU crop/write and does not require CUDA devices
- Audio cache uses `ffmpeg` plus `python_speech_features.logfbank` on CPU
- Training uses frozen AV-HuBERT backbone plus `Linear(1024, 1)` head on GPU-only entrypoints
- Outputs include `best_head.pt`, `last_head.pt`, `summary.json`, `training_curves.png`, `train.log`, and copied `config.yaml`

- [ ] **Step 2: Verify dataset-specific default values**

Check the three classifier configs and six preprocess configs and note:

- AV1M epochs `10`, batch size `4`, num workers `8`
- FakeAVCeleb epochs `10`, batch size `4`, num workers `8`
- MAVOS-DD epochs `100`, batch size `4`, num workers `8`
- Detect runtime `devices=[0,1]`, `workers_per_device=3`, `cpu_threads_per_worker=8`
- Align runtime `devices=[]`, `workers_per_device=6`, `cpu_threads_per_worker=8`
- Audio cache `num_procs=12`, `cpu_threads_per_worker=4`, `stack_order_audio=4`

- [ ] **Step 3: Use these facts as the only allowed description source**

Do not infer new behavior. If a detail is not supported by code or config, omit it.

### Task 2: Add Detailed Chinese Comments To All Configs

**Files:**
- Modify: `configs/avhubert_classifier.yaml`
- Modify: `configs/avhubert_preprocess.yaml`
- Modify: `configs/avhubert_preprocess_align.yaml`
- Modify: `configs/fakeavceleb_classifier.yaml`
- Modify: `configs/fakeavceleb_preprocess.yaml`
- Modify: `configs/fakeavceleb_preprocess_align.yaml`
- Modify: `configs/mavos_dd_real_fullfake_classifier.yaml`
- Modify: `configs/mavos_dd_real_fullfake_preprocess.yaml`
- Modify: `configs/mavos_dd_real_fullfake_preprocess_align.yaml`

- [ ] **Step 1: Add section-level comments**

For every top-level section, add a short Chinese comment explaining:

- `paths`: where each stage reads inputs and writes artifacts
- `preprocess`: which preprocess stage runs and what crop/alignment knobs mean
- `runtime`: how many workers/devices/threads are allocated
- `data`: how training-time batching and transforms behave
- `model`: how the frozen backbone and linear probe assumptions work
- `train`: what hardware/process settings and optimization settings control
- `audio_cache`: how CPU-side feature extraction is parallelized
- `logging` and `visualization`: what files are written

- [ ] **Step 2: Add key-field comments without changing values**

Annotate important fields such as:

- `raw_video_root`, `split_dir`, `artifact_root`, `mouth_roi_root`, `audio_feature_root`, `output_root`
- `stage`, `manifest_name`, `detector_batch_size`, `workers_per_device`, `cpu_threads_per_worker`
- `max_frames`, `image_crop_size`, `horizontal_flip_prob`
- `devices`, `backend`, `epochs`, `batch_size`, `num_workers`, `amp`
- `num_procs`, `stack_order_audio`

Do not reorder fields aggressively and do not change any values.

- [ ] **Step 3: Keep comments concise and operational**

Comments should explain why an operator cares, for example:

- whether a path is input, intermediate, or final output
- whether a field affects GPU, CPU, or both
- whether a value is per-GPU, per-process, or global

### Task 3: Write The Framework Document

**Files:**
- Create: `docs/training_framework.md`

- [ ] **Step 1: Write the document header and scope**

Start with:

- project scope
- shared framework versus dataset-specific differences
- current default hardware and environments

- [ ] **Step 2: Add the end-to-end pipeline sections**

Document these stages in order:

1. Split building
2. Manifest generation
3. Detect stage
4. Align stage
5. Audio cache
6. Dataset loading and collate
7. Backbone loading and linear probe
8. Train/val/test loop
9. Output artifacts and where they are written

For each stage, include:

- command or entrypoint
- input files/directories
- output files/directories
- CPU/GPU attribution
- required environment

- [ ] **Step 3: Add the default configuration summary**

Include a section that lists the nine configs and their roles, plus the default values that matter operationally.

### Task 4: Validate And Update Project Notes

**Files:**
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

- [ ] **Step 1: Run YAML parse verification**

Run:

```bash
python - <<'PY'
from pathlib import Path
from src.utils.project import load_config
for path in sorted(Path("configs").glob("*.yaml")):
    cfg = load_config(path)
    print(path.as_posix(), list(cfg.keys()))
PY
```

Expected: all nine config files load successfully.

- [ ] **Step 2: Run lightweight entrypoint verification**

Run:

```bash
python scripts/preprocess_av1m_mouth_roi.py --help
python scripts/preprocess_fakeavceleb.py --help
python scripts/preprocess_mavos_dd_real_fullfake.py --help
python scripts/train_fakeavceleb.py --help
python scripts/train_mavos_dd_real_fullfake.py --help
```

Expected: help output prints normally.

- [ ] **Step 3: Update notes and monthly log**

Record:

- new document path
- YAML comment coverage
- verification commands and results

- [ ] **Step 4: Commit**

```bash
git add docs/training_framework.md docs/notes.md docs/logs/2026-04.md configs/*.yaml docs/superpowers/plans/2026-04-20-training-framework-docs-and-yaml-comments.md
git commit -m "docs: document training framework and annotate configs"
```
