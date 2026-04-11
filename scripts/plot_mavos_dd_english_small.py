from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.training_curves import plot_training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MAVOS-DD English small training curves from summary.json")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or args.summary.with_name("training_curves.png")
    result = plot_training_summary(args.summary, output_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
