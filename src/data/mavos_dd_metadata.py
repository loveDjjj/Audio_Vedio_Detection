from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def find_arrow_file(metadata_root: Path) -> Path:
    candidates = sorted(metadata_root.glob("data-*.arrow"))
    if not candidates:
        raise FileNotFoundError(f"No metadata arrow file found under {metadata_root}")
    return candidates[0]


def load_mavos_dd_records(metadata_root: Path) -> list[dict]:
    from datasets import Dataset

    arrow_path = find_arrow_file(metadata_root)
    dataset = Dataset.from_file(str(arrow_path))
    return [dataset[index] for index in range(len(dataset))]


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _sorted_nested_counter(counter_map: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {str(key): _sorted_counter(counter_map[key]) for key in sorted(counter_map)}


def summarize_mavos_dd_records(records: Iterable[dict]) -> dict:
    total_records = 0
    label_counter: Counter = Counter()
    split_counter: Counter = Counter()
    language_counter: Counter = Counter()
    generator_counter: Counter = Counter()
    open_set_model_counter: Counter = Counter()
    open_set_language_counter: Counter = Counter()
    split_language_counter: dict[str, Counter] = defaultdict(Counter)
    split_generator_counter: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        total_records += 1
        label = str(record["label"])
        split = str(record["split"])
        language = str(record["language"])
        generator = str(record["generative_method"])
        open_set_model = bool(record["open_set_model"])
        open_set_language = bool(record["open_set_language"])

        label_counter[label] += 1
        split_counter[split] += 1
        language_counter[language] += 1
        generator_counter[generator] += 1
        open_set_model_counter[str(open_set_model).lower()] += 1
        open_set_language_counter[str(open_set_language).lower()] += 1
        split_language_counter[split][language] += 1
        split_generator_counter[split][generator] += 1

    return {
        "total_records": total_records,
        "unique_languages": len(language_counter),
        "unique_generative_methods": len(generator_counter),
        "labels": _sorted_counter(label_counter),
        "splits": _sorted_counter(split_counter),
        "languages": _sorted_counter(language_counter),
        "generative_methods": _sorted_counter(generator_counter),
        "open_set_model": _sorted_counter(open_set_model_counter),
        "open_set_language": _sorted_counter(open_set_language_counter),
        "split_language": _sorted_nested_counter(split_language_counter),
        "split_generative_method": _sorted_nested_counter(split_generator_counter),
    }


def write_mavos_dd_summary(summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
