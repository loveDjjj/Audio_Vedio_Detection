from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_relative_paths(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["relative_path"] for row in reader]


def _to_manifest_ids(relative_paths: list[str]) -> list[str]:
    manifest_ids: list[str] = []
    for relative_path in relative_paths:
        path = Path(relative_path)
        if path.suffix != ".mp4":
            raise ValueError(f"Expected an .mp4 path, got: {relative_path}")
        manifest_ids.append(path.with_suffix("").as_posix())
    return manifest_ids


def build_manifests(
    split_dir: Path,
    output_dir: Path,
    split_names: list[str] | tuple[str, ...] = ("train", "val", "test"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {"splits": {}}
    all_ids: list[str] = []

    for split_name in split_names:
        csv_path = split_dir / f"{split_name}.csv"
        relative_paths = _load_relative_paths(csv_path)
        manifest_ids = _to_manifest_ids(relative_paths)
        all_ids.extend(manifest_ids)

        manifest_path = output_dir / f"{split_name}.list"
        manifest_path.write_text("\n".join(manifest_ids) + "\n", encoding="utf-8")

        summary["splits"][split_name] = {
            "videos": len(relative_paths),
            "manifest_entries": len(manifest_ids),
        }

    unique_all_ids = sorted(set(all_ids))
    (output_dir / "all.list").write_text(
        "\n".join(unique_all_ids) + "\n",
        encoding="utf-8",
    )
    summary["all"] = {
        "videos": len(all_ids),
        "unique_manifest_entries": len(unique_all_ids),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
