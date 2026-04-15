# FakeAVCeleb Independent Training Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable FakeAVCeleb `RealVideo-RealAudio` vs `FakeVideo-FakeAudio` AV-HuBERT branch with deterministic split generation, dataset-specific YAML configs, thin wrappers, generated split artifacts, and minimal regression coverage.

**Architecture:** Keep the current pipeline unchanged after split generation. Implement FakeAVCeleb-specific metadata parsing and balanced split sampling in a focused helper module plus a thin CLI builder script, then wire the existing preprocess, audio-cache, and training runtimes through dataset-specific YAML files and wrapper scripts. Track the addition in repository docs and generate committed split CSVs under `splits/fakeavceleb_real_fullfake`.

**Tech Stack:** Python 3, `csv`, `json`, `pathlib`, `random`, `unittest`, YAML configs, existing AV-HuBERT preprocessing/training scripts.

---

## File Structure

**Create**

- `src/data/fakeavceleb_subset.py`
- `dataset/build_fakeavceleb_real_fullfake_splits.py`
- `configs/fakeavceleb_preprocess.yaml`
- `configs/fakeavceleb_classifier.yaml`
- `scripts/preprocess_fakeavceleb.py`
- `scripts/cache_fakeavceleb_audio_features.py`
- `scripts/train_fakeavceleb.py`
- `tests/test_fakeavceleb_subset.py`
- `tests/test_fakeavceleb_wrappers.py`

**Modify**

- `README.md`
- `docs/notes.md`
- `docs/logs/2026-04.md`

**Generate**

- `splits/fakeavceleb_real_fullfake/train.csv`
- `splits/fakeavceleb_real_fullfake/val.csv`
- `splits/fakeavceleb_real_fullfake/test.csv`
- `splits/fakeavceleb_real_fullfake/summary.json`

## Task 1: Build FakeAVCeleb Metadata Parser And Split Generator

**Files:**

- Create: `src/data/fakeavceleb_subset.py`
- Create: `dataset/build_fakeavceleb_real_fullfake_splits.py`
- Create: `tests/test_fakeavceleb_subset.py`

- [ ] **Step 1: Write the failing subset-builder tests**

Create `tests/test_fakeavceleb_subset.py` with coverage for:

- parsing the malformed `meta_data.csv` trailing empty column
- keeping only `RealVideo-RealAudio` and `FakeVideo-FakeAudio`
- building repository-relative `relative_path`
- deterministic `1:1` real/fake balancing
- deterministic `train/val/test = 0.8/0.1/0.1` splitting
- method-aware fake sampling

```python
from pathlib import Path
import tempfile
import unittest

from src.data.fakeavceleb_subset import (
    FakeAVCelebRecord,
    build_split_rows,
    load_fakeavceleb_records,
    sample_balanced_binary_records,
    split_records,
)


class FakeAVCelebSubsetTest(unittest.TestCase):
    def _write_metadata(self, root: Path) -> None:
        metadata = """source,target1,target2,method,category,type,race,gender,path,
id00001,-,-,real,A,RealVideo-RealAudio,African,men,00001.mp4,FakeAVCeleb/RealVideo-RealAudio/African/men/id00001
id00002,-,-,real,A,RealVideo-RealAudio,African,men,00002.mp4,FakeAVCeleb/RealVideo-RealAudio/African/men/id00002
id00003,id10001,4,wav2lip,D,FakeVideo-FakeAudio,African,men,00003_4_id10001_wavtolip.mp4,FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00003
id00004,id10002,8,fsgan-wav2lip,D,FakeVideo-FakeAudio,African,men,00004_8_id10002_fsgan-wav2lip.mp4,FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00004
id00005,id10003,-,faceswap-wav2lip,D,FakeVideo-FakeAudio,African,women,00005_id10003_faceswap-wav2lip.mp4,FakeAVCeleb/FakeVideo-FakeAudio/African/women/id00005
id99999,-,-,rtvc,B,RealVideo-FakeAudio,African,men,ignore.mp4,FakeAVCeleb/RealVideo-FakeAudio/African/men/id99999
"""
        (root / "meta_data.csv").write_text(metadata, encoding="utf-8")

    def _touch_video(self, root: Path, relative_path: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")

    def test_load_fakeavceleb_records_keeps_only_a_and_d(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_metadata(root)
            self._touch_video(root, "RealVideo-RealAudio/African/men/id00001/00001.mp4")
            self._touch_video(root, "RealVideo-RealAudio/African/men/id00002/00002.mp4")
            self._touch_video(root, "FakeVideo-FakeAudio/African/men/id00003/00003_4_id10001_wavtolip.mp4")
            self._touch_video(root, "FakeVideo-FakeAudio/African/men/id00004/00004_8_id10002_fsgan-wav2lip.mp4")
            self._touch_video(root, "FakeVideo-FakeAudio/African/women/id00005/00005_id10003_faceswap-wav2lip.mp4")

            records = load_fakeavceleb_records(root)

        self.assertEqual(len(records), 5)
        self.assertEqual({record.type for record in records}, {"RealVideo-RealAudio", "FakeVideo-FakeAudio"})
        self.assertEqual(
            records[0].relative_path,
            "RealVideo-RealAudio/African/men/id00001/00001.mp4",
        )
```

- [ ] **Step 2: Run the new subset tests and verify they fail**

Run:

```bash
python -m unittest tests.test_fakeavceleb_subset -v
```

Expected:

```text
ERROR: Failed to import test module: test_fakeavceleb_subset
ModuleNotFoundError: No module named 'src.data.fakeavceleb_subset'
```

- [ ] **Step 3: Implement the focused FakeAVCeleb subset helper**

Create `src/data/fakeavceleb_subset.py` with a single responsibility: parse metadata rows, build validated records, balance real/fake samples, split deterministically, and write rows for CSV/summary generation.

```python
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FakeAVCelebRecord:
    relative_path: str
    label: int
    label_name: str
    source: str
    target1: str
    target2: str
    method: str
    category: str
    type: str
    race: str
    gender: str
    filename: str
```

- [ ] **Step 4: Implement the CLI split-builder script**

Create `dataset/build_fakeavceleb_real_fullfake_splits.py` as a thin entrypoint around the helper module.

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 5: Run the subset tests and a local split smoke test**

Run:

```bash
python -m unittest tests.test_fakeavceleb_subset -v
python dataset/build_fakeavceleb_real_fullfake_splits.py --root dataset/FakeAVCeleb --output-dir .tmp_fakeavceleb_split
```

- [ ] **Step 6: Commit the split-builder work**

```bash
git add tests/test_fakeavceleb_subset.py src/data/fakeavceleb_subset.py dataset/build_fakeavceleb_real_fullfake_splits.py
git commit -m "feat: add fakeavceleb split builder"
```

## Task 2: Add FakeAVCeleb Configs And Wrapper Scripts

**Files:**

- Create: `configs/fakeavceleb_preprocess.yaml`
- Create: `configs/fakeavceleb_classifier.yaml`
- Create: `scripts/preprocess_fakeavceleb.py`
- Create: `scripts/cache_fakeavceleb_audio_features.py`
- Create: `scripts/train_fakeavceleb.py`
- Create: `tests/test_fakeavceleb_wrappers.py`

- [ ] **Step 1: Write failing wrapper/config tests**

Create `tests/test_fakeavceleb_wrappers.py` to verify:

- preprocess wrapper points to `configs/fakeavceleb_preprocess.yaml`
- audio-cache wrapper points to `configs/fakeavceleb_classifier.yaml`
- training wrapper forwards the config to `scripts/train_avhubert_classifier.py`
- YAML files resolve the FakeAVCeleb-specific paths

```python
from pathlib import Path
import runpy
import unittest
from unittest import mock

from src.utils.project import load_config


class FakeAVCelebWrapperTest(unittest.TestCase):
    def test_fakeavceleb_preprocess_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_preprocess.yaml"))
        self.assertEqual(config["paths"]["raw_video_root"], "dataset/FakeAVCeleb")
        self.assertEqual(config["paths"]["split_dir"], "splits/fakeavceleb_real_fullfake")

    def test_fakeavceleb_classifier_config_paths(self) -> None:
        config = load_config(Path("configs/fakeavceleb_classifier.yaml"))
        self.assertEqual(config["paths"]["raw_video_root"], "dataset/FakeAVCeleb")
        self.assertEqual(config["paths"]["output_root"], "outputs/avhubert/fakeavceleb_real_fullfake")
```

- [ ] **Step 2: Run the wrapper tests and verify they fail**

Run:

```bash
python -m unittest tests.test_fakeavceleb_wrappers -v
```

Expected:

```text
FileNotFoundError: ... configs/fakeavceleb_preprocess.yaml
```

- [ ] **Step 3: Add the FakeAVCeleb preprocess config**

Create `configs/fakeavceleb_preprocess.yaml` by cloning the MAVOS-DD preprocess shape and only switching the dataset-specific paths.

```yaml
paths:
  split_dir: splits/fakeavceleb_real_fullfake
  raw_video_root: dataset/FakeAVCeleb
  artifact_root: artifacts/avhubert/fakeavceleb_real_fullfake
  manifest_dir: artifacts/avhubert/fakeavceleb_real_fullfake/manifests
  landmark_dir: artifacts/avhubert/fakeavceleb_real_fullfake/landmarks
  mouth_roi_root: artifacts/avhubert/fakeavceleb_real_fullfake/mouth_roi
  cnn_detector_path: resources/dlib/mmod_human_face_detector.dat
  face_predictor_path: resources/dlib/shape_predictor_68_face_landmarks.dat
  mean_face_path: resources/avhubert/20words_mean_face.npy
```

- [ ] **Step 4: Add the FakeAVCeleb classifier config**

Create `configs/fakeavceleb_classifier.yaml` by cloning the current AV1M classifier shape and only switching the dataset-specific paths and experiment name.

```yaml
experiment_name: fakeavceleb_real_fullfake_avhubert

paths:
  dataset_root: dataset/FakeAVCeleb
  raw_video_root: dataset/FakeAVCeleb
  split_dir: splits/fakeavceleb_real_fullfake
  artifact_root: artifacts/avhubert/fakeavceleb_real_fullfake
  output_root: outputs/avhubert/fakeavceleb_real_fullfake
```

- [ ] **Step 5: Add the FakeAVCeleb wrapper scripts**

Create thin wrappers matching the existing MAVOS-DD pattern.

```python
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 6: Run the wrapper/config tests and help smoke checks**

Run:

```bash
python -m unittest tests.test_fakeavceleb_wrappers -v
python scripts/train_fakeavceleb.py --help
```

- [ ] **Step 7: Commit the config/wrapper work**

```bash
git add tests/test_fakeavceleb_wrappers.py configs/fakeavceleb_preprocess.yaml configs/fakeavceleb_classifier.yaml scripts/preprocess_fakeavceleb.py scripts/cache_fakeavceleb_audio_features.py scripts/train_fakeavceleb.py
git commit -m "feat: add fakeavceleb configs and wrappers"
```

## Task 3: Generate Committed Split Artifacts And Update Repository Docs

**Files:**

- Generate: `splits/fakeavceleb_real_fullfake/train.csv`
- Generate: `splits/fakeavceleb_real_fullfake/val.csv`
- Generate: `splits/fakeavceleb_real_fullfake/test.csv`
- Generate: `splits/fakeavceleb_real_fullfake/summary.json`
- Modify: `README.md`
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

- [ ] **Step 1: Generate the real FakeAVCeleb split artifacts**

Run:

```bash
python dataset/build_fakeavceleb_real_fullfake_splits.py --root dataset/FakeAVCeleb --output-dir splits/fakeavceleb_real_fullfake --seed 42
```

- [ ] **Step 2: Verify the generated split files contain the expected schema**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
for name in ["train", "val", "test"]:
    path = Path("splits/fakeavceleb_real_fullfake") / f"{name}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    print(name, sorted(row.keys()))
PY
```

- [ ] **Step 3: Update the project docs to mention the new FakeAVCeleb branch**

Modify `README.md` to add the new split-builder command, the FakeAVCeleb preprocess/audio-cache/train commands, and the new config/output directories.

- [ ] **Step 4: Run the final verification commands**

Run:

```bash
python -m unittest tests.test_fakeavceleb_subset tests.test_fakeavceleb_wrappers -v
python dataset/build_fakeavceleb_real_fullfake_splits.py --root dataset/FakeAVCeleb --output-dir .tmp_fakeavceleb_verify --seed 42
python scripts/train_fakeavceleb.py --help
```

- [ ] **Step 5: Commit the generated splits and docs**

```bash
git add dataset/build_fakeavceleb_real_fullfake_splits.py src/data/fakeavceleb_subset.py configs/fakeavceleb_preprocess.yaml configs/fakeavceleb_classifier.yaml scripts/preprocess_fakeavceleb.py scripts/cache_fakeavceleb_audio_features.py scripts/train_fakeavceleb.py tests/test_fakeavceleb_subset.py tests/test_fakeavceleb_wrappers.py splits/fakeavceleb_real_fullfake README.md docs/notes.md docs/logs/2026-04.md
git commit -m "feat: add fakeavceleb real fullfake pipeline"
```

## Self-Review

### Spec coverage

- Data scope limited to `RealVideo-RealAudio` and `FakeVideo-FakeAudio`: covered in Task 1 helper logic and generated splits.
- Random video-level split: covered in Task 1 `split_records()`.
- `1:1` balanced subset: covered in Task 1 `sample_balanced_binary_records()`.
- Fake-method-aware sampling: covered in Task 1 `_sample_fake_records()`.
- Dataset-specific preprocess/classifier YAML and wrappers: covered in Task 2.
- Committed split artifacts and docs updates: covered in Task 3.

### Placeholder scan

- No deferred placeholder markers remain in the plan.
- Commands, paths, and expected outputs are present for each task.

### Type consistency

- The plan uses `FakeAVCelebRecord` consistently across parser, sampler, splitter, and CSV writer code.
- All config paths use the same dataset branch name: `fakeavceleb_real_fullfake`.
- Wrapper scripts consistently point to `configs/fakeavceleb_preprocess.yaml` and `configs/fakeavceleb_classifier.yaml`.
