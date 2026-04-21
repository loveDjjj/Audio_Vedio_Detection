# Notes

## Request
Document the current training framework and preprocess framework in a dedicated Markdown file, and add detailed Chinese comments to every config under `configs/*.yaml`.

The required scope is:

1. Explain the end-to-end preprocess and training pipeline step by step.
2. Explain which stages mainly run on CPU and which mainly run on GPU.
3. Explain required environments and runtime dependencies.
4. Explain the inputs and outputs of each stage.
5. Explain the current default configuration sizes.
6. Add detailed Chinese comments to all 9 YAML files under `configs/`.

## Modified Files

- configs/avhubert_classifier.yaml
- configs/avhubert_preprocess.yaml
- configs/avhubert_preprocess_align.yaml
- configs/fakeavceleb_classifier.yaml
- configs/fakeavceleb_preprocess.yaml
- configs/fakeavceleb_preprocess_align.yaml
- configs/mavos_dd_real_fullfake_classifier.yaml
- configs/mavos_dd_real_fullfake_preprocess.yaml
- configs/mavos_dd_real_fullfake_preprocess_align.yaml
- docs/training_framework.md
- docs/superpowers/plans/2026-04-20-training-framework-docs-and-yaml-comments.md
- docs/notes.md
- docs/logs/2026-04.md

## Changes

- Added detailed Chinese comments to all 9 config files under `configs/`, covering section purpose, path roles, CPU/GPU placement, per-process/per-GPU semantics, and default value meaning.
- Added `docs/training_framework.md` to document the shared split -> manifest -> detect -> align -> audio cache -> dataset -> frozen AV-HuBERT linear probe -> train/val/test pipeline.
- Documented the current server environment split in the framework doc: `oneday` for preprocess, `avhubert` for training, and the current server caveat that audio cache has been validated in `oneday` because it contains `python_speech_features`.
- Summarized each stage's inputs, outputs, compute placement, and default configuration values in one place.
- Wrote an implementation plan file under `docs/superpowers/plans/` before making the changes.

## Verification

```bash
python - <<'PY'
from pathlib import Path
from src.utils.project import load_config
for path in sorted(Path("configs").glob("*.yaml")):
    cfg = load_config(path)
    print(path.as_posix(), sorted(cfg.keys()))
PY

python scripts/preprocess_av1m_mouth_roi.py --help
python scripts/preprocess_fakeavceleb.py --help
python scripts/preprocess_mavos_dd_real_fullfake.py --help
python scripts/train_fakeavceleb.py --help
python scripts/train_mavos_dd_real_fullfake.py --help
```

Result: pass. All 9 config files load successfully, and the main preprocess/train wrapper entrypoints still print help normally after the YAML comment updates.

## Git

- branch: `main`
- commit: `docs: document training framework and annotate configs`
