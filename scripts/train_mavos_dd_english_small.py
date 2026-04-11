from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAVOS-DD English small training with the dedicated config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mavos_dd_english_small_classifier.yaml"),
        help="Path to the MAVOS-DD English small training config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [sys.executable, "scripts/train_avhubert_classifier.py", "--config", str(args.config)]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
