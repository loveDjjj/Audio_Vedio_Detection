from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.manifest_builder import build_manifests
from src.utils.project import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AV-HuBERT mouth-ROI manifest lists from current CSV splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/avhubert_classifier.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    split_dir = resolve_path(config["paths"]["split_dir"])
    manifest_dir = resolve_path(config["paths"]["manifest_dir"])
    split_names = config["preprocess"]["manifest_names"]

    summary = build_manifests(split_dir=split_dir, output_dir=manifest_dir, split_names=split_names)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
