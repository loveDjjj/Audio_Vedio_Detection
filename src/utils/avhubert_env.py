from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_avhubert_repo(repo_root: str | Path) -> Path:
    repo_path = Path(repo_root).resolve()
    fairseq_path = repo_path / "fairseq"
    avhubert_path = repo_path / "avhubert"

    if not repo_path.exists():
        raise FileNotFoundError(
            f"AV-HuBERT repo not found: {repo_path}. "
            "Clone facebookresearch/av_hubert into third_party/av_hubert first."
        )
    if not fairseq_path.exists():
        raise FileNotFoundError(
            f"AV-HuBERT fairseq submodule not found: {fairseq_path}. "
            "Run `git submodule update --init --recursive` inside third_party/av_hubert."
        )

    # Upstream AV-HuBERT switches to a top-level "debug" import path when argv has
    # only the script name, which breaks our normal package imports.
    if len(sys.argv) == 1:
        sys.argv.append("--avhubert-import")

    # AV-HuBERT uses both package-relative and top-level imports depending on argv shape.
    for path in (fairseq_path, avhubert_path, repo_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_path


def import_avhubert_modules(repo_root: str | Path):
    bootstrap_avhubert_repo(repo_root)

    import fairseq  # type: ignore
    from avhubert import hubert  # type: ignore
    from avhubert import hubert_asr  # type: ignore
    from avhubert import hubert_pretraining  # type: ignore

    return fairseq, hubert_pretraining, hubert, hubert_asr
