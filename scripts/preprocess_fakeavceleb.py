from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.runtime import run_preprocess_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FakeAVCeleb mouth ROI preprocessing from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fakeavceleb_preprocess.yaml"),
        help="Path to the preprocess YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_preprocess_from_config(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
