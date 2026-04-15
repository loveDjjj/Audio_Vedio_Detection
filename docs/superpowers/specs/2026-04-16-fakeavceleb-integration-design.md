# FakeAVCeleb Independent Training Branch Design

## Goal

Add FakeAVCeleb to the current AV-HuBERT pipeline as an independent binary classification branch, without changing the existing AV-Deepfake1M or MAVOS-DD flows.

This branch only uses the following FakeAVCeleb categories under `dataset/FakeAVCeleb`:

- `RealVideo-RealAudio`
- `FakeVideo-FakeAudio`

The branch keeps the current repository architecture:

1. build split CSVs
2. preprocess mouth ROI from raw mp4
3. cache audio features from raw mp4
4. train the frozen AV-HuBERT linear probe

## Chosen Scope

### Included data

- Positive class: `FakeVideo-FakeAudio`
- Negative class: `RealVideo-RealAudio`

### Excluded data

- `RealVideo-FakeAudio`
- `FakeVideo-RealAudio`

These excluded categories remain untouched in the repository and are not used by the first FakeAVCeleb branch.

## Why This Design

The current training and preprocessing code is already close to dataset-agnostic after split generation.

- The training entrypoint reads `train.csv`, `val.csv`, and `test.csv` from the configured split directory and builds the dataset from `relative_path` plus `label`.
- The preprocessing runtime converts split CSV rows into manifest ids by reading the `relative_path` column and stripping `.mp4`.
- The audio cache runtime also reads only the `relative_path` column and resolves each raw video under the configured `raw_video_root`.

This means FakeAVCeleb should be integrated by adding a new dataset-specific split builder and a new set of dataset-specific YAML configs and wrapper scripts, rather than by importing FakeAVCeleb's official training code.

## Local Data Assumptions

The current local FakeAVCeleb layout is:

- `dataset/FakeAVCeleb/RealVideo-RealAudio/.../*.mp4`
- `dataset/FakeAVCeleb/FakeVideo-FakeAudio/.../*.mp4`
- `dataset/FakeAVCeleb/meta_data.csv`
- `dataset/FakeAVCeleb/README.txt`

Observed properties from the current local copy:

- `RealVideo-RealAudio`: 500 mp4 files
- `FakeVideo-FakeAudio`: 10857 mp4 files
- all selected metadata rows currently resolve to existing local files

`meta_data.csv` contains a trailing empty column in the header, so the split builder must parse it with `csv.reader` and map columns manually instead of trusting the default `DictReader` header names.

## Label Mapping

The branch uses binary labels:

- `RealVideo-RealAudio -> 0`
- `FakeVideo-FakeAudio -> 1`

The split CSV should still preserve useful metadata columns for later analysis:

- `relative_path`
- `label`
- `label_name`
- `source`
- `target1`
- `target2`
- `method`
- `category`
- `type`
- `race`
- `gender`
- `filename`

Only `relative_path` and `label` are required by the current reusable pipeline, but the extra fields make later error analysis possible without rebuilding the splits.

## Split Strategy

### Chosen strategy

Use random video-level splitting.

### Ratios

- train: 0.8
- val: 0.1
- test: 0.1

### Class balancing

The first version should use balanced sampling before splitting:

- keep all 500 real videos
- sample 500 fake videos
- combine them into a 1000-video binary dataset
- then apply random video-level train/val/test splitting

This keeps the first branch simple and avoids changing the current trainer, which does not yet include class-weighted loss or balanced sampling.

### Fake-method coverage

Fake videos should be sampled in a method-aware way so that the selected 500 fake samples are not dominated by a single generator.

Preferred behavior:

1. group `FakeVideo-FakeAudio` rows by `method`
2. allocate a proportional sample count per method
3. use deterministic seeded random sampling inside each method
4. merge and sort the selected rows

If proportional rounding leaves a shortfall or overflow, the script should reconcile the difference deterministically.

## Repository Additions

### New split builder

Add:

- `dataset/build_fakeavceleb_real_fullfake_splits.py`

Responsibilities:

1. parse `dataset/FakeAVCeleb/meta_data.csv`
2. keep only `RealVideo-RealAudio` and `FakeVideo-FakeAudio`
3. resolve each row into a repository-relative mp4 path
4. verify the target mp4 exists under `dataset/FakeAVCeleb`
5. apply 1:1 balanced selection with deterministic sampling
6. randomly split into `train/val/test`
7. write split CSVs and `summary.json`

### New split directory

Add:

- `splits/fakeavceleb_real_fullfake/train.csv`
- `splits/fakeavceleb_real_fullfake/val.csv`
- `splits/fakeavceleb_real_fullfake/test.csv`
- `splits/fakeavceleb_real_fullfake/summary.json`

### New preprocess config

Add:

- `configs/fakeavceleb_preprocess.yaml`

Key path decisions:

- `raw_video_root: dataset/FakeAVCeleb`
- `split_dir: splits/fakeavceleb_real_fullfake`
- `artifact_root: artifacts/avhubert/fakeavceleb_real_fullfake`
- `manifest_dir: artifacts/avhubert/fakeavceleb_real_fullfake/manifests`
- `landmark_dir: artifacts/avhubert/fakeavceleb_real_fullfake/landmarks`
- `mouth_roi_root: artifacts/avhubert/fakeavceleb_real_fullfake/mouth_roi`

Other preprocessing parameters should reuse the current AV-HuBERT defaults unless a dataset-specific issue is observed.

### New classifier config

Add:

- `configs/fakeavceleb_classifier.yaml`

Key path decisions:

- `dataset_root: dataset/FakeAVCeleb`
- `raw_video_root: dataset/FakeAVCeleb`
- `split_dir: splits/fakeavceleb_real_fullfake`
- `artifact_root: artifacts/avhubert/fakeavceleb_real_fullfake`
- `mouth_roi_root: artifacts/avhubert/fakeavceleb_real_fullfake/mouth_roi`
- `audio_feature_root: artifacts/avhubert/fakeavceleb_real_fullfake/audio_features`
- `output_root: outputs/avhubert/fakeavceleb_real_fullfake`

Training hyperparameters should start from the current AV1M defaults and only be adjusted after the first dry run if resource pressure requires it.

### New thin wrapper scripts

Add:

- `scripts/preprocess_fakeavceleb.py`
- `scripts/cache_fakeavceleb_audio_features.py`
- `scripts/train_fakeavceleb.py`

These should follow the same wrapper pattern already used for MAVOS-DD and only point to the FakeAVCeleb-specific configs.

## Data Flow

The final FakeAVCeleb branch should run as:

```bash
python dataset/build_fakeavceleb_real_fullfake_splits.py
python scripts/preprocess_fakeavceleb.py
python scripts/cache_fakeavceleb_audio_features.py
python scripts/train_fakeavceleb.py
```

The generated files should flow through these locations:

- split CSVs: `splits/fakeavceleb_real_fullfake/`
- manifests: `artifacts/avhubert/fakeavceleb_real_fullfake/manifests/`
- landmarks: `artifacts/avhubert/fakeavceleb_real_fullfake/landmarks/`
- mouth ROI: `artifacts/avhubert/fakeavceleb_real_fullfake/mouth_roi/`
- audio features: `artifacts/avhubert/fakeavceleb_real_fullfake/audio_features/`
- training outputs: `outputs/avhubert/fakeavceleb_real_fullfake/<timestamp>/`

## CSV Format

Each split CSV should use a superset of the current common format. Example columns:

```text
relative_path,label,label_name,source,target1,target2,method,category,type,race,gender,filename
RealVideo-RealAudio/African/men/id00076/00109.mp4,0,real,id00076,-,-,real,A,RealVideo-RealAudio,African,men,00109.mp4
FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.mp4,1,fake,id00076,id00476,10,wav2lip,D,FakeVideo-FakeAudio,African,men,00109_10_id00476_wavtolip.mp4
```

Exact `target1/target2/method` values must come from the parsed metadata row, not from ad hoc filename guessing.

## Non-Goals

The first FakeAVCeleb branch does not include:

- joint training with AV-Deepfake1M or MAVOS-DD
- support for categories `B` and `C`
- class-weighted BCE or balanced samplers in the trainer
- identity-level leakage control
- reuse of the official FakeAVCeleb `FRAMES_PNG` or `SPECTROGRAM` training path

These can be added later if the first branch is stable.

## Risks And Constraints

### Class imbalance

The raw selected data is highly imbalanced:

- real: 500
- fake: 10857

Without balancing, the current trainer would likely be biased because it uses plain `BCEWithLogitsLoss()` with no explicit class weighting.

### Random video split leakage

The chosen split mode is random video-level splitting. This is acceptable for the first branch because it matches the requested design, but it is weaker than identity-level or source-level splitting and may yield optimistic test performance.

### Metadata parsing edge case

Because `meta_data.csv` has a malformed trailing empty header cell, the split builder must not rely on naive header mapping.

### Dataset-specific compute pressure

Even after 1:1 balancing, the branch still requires:

- dlib CUDA preprocessing resources
- ffmpeg for audio extraction
- CUDA for training

The current repository assumptions on these requirements remain unchanged.

## Implementation Outline

When implementation starts, the work should be done in this order:

1. build and validate the FakeAVCeleb split generator
2. add FakeAVCeleb preprocess and classifier YAML files
3. add thin preprocess/audio-cache/train wrappers
4. run split-generation verification
5. run a small preprocess smoke test
6. run an audio-cache smoke test
7. run training entrypoint help or a short smoke test

## Acceptance Criteria

The FakeAVCeleb branch is considered integrated when:

1. `dataset/build_fakeavceleb_real_fullfake_splits.py` generates deterministic `train/val/test.csv` plus `summary.json`
2. the generated CSVs use valid `relative_path` values under `dataset/FakeAVCeleb`
3. mouth ROI preprocessing runs using the new FakeAVCeleb preprocess config
4. audio feature caching runs using the new FakeAVCeleb classifier config
5. `scripts/train_fakeavceleb.py` launches the existing AV-HuBERT binary training entrypoint with the new config
