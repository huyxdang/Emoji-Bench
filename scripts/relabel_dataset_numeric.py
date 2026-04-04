#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.dataset import DatasetManifest, build_dataset_card
from emoji_bench.formatter import system_from_json, system_to_json
from emoji_bench.numeric_labels import (
    build_two_digit_symbol_map,
    relabel_system,
    relabel_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite an existing Emoji-Bench dataset with random two-digit numeric symbols.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing manifest.json and split JSONL files.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the relabeled dataset should be written.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Override the output dataset name. Defaults to '<input-name>-numbers'.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to each system seed before sampling two-digit labels.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest(**manifest_data)
    output_dataset_name = args.dataset_name or f"{manifest.dataset_name}-numbers"

    output_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, tuple[dict, str]] = {}
    for split in ("train", "validation", "test"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue

        output_path = output_dir / f"{split}.jsonl"
        with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
            for line in src:
                record = json.loads(line)
                system_json = record["system_json"]

                if system_json not in cache:
                    system = system_from_json(system_json)
                    mapping_seed = int(record.get("system_seed", system.seed)) + args.seed_offset
                    symbol_map = build_two_digit_symbol_map(system.symbols, seed=mapping_seed)
                    relabeled_system = relabel_system(system, symbol_map)
                    cache[system_json] = (symbol_map, system_to_json(relabeled_system))

                symbol_map, relabeled_system_json = cache[system_json]
                relabeled_record = dict(record)
                relabeled_record["prompt"] = relabel_text(record["prompt"], symbol_map)
                relabeled_record["system_json"] = relabeled_system_json
                relabeled_record["example_id"] = _rewrite_example_id(
                    record["example_id"],
                    source_dataset_name=manifest.dataset_name,
                    target_dataset_name=output_dataset_name,
                )
                dst.write(json.dumps(relabeled_record, ensure_ascii=False) + "\n")

    output_manifest = replace(manifest, dataset_name=output_dataset_name)
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        build_dataset_card(output_manifest),
        encoding="utf-8",
    )


def _rewrite_example_id(
    example_id: str,
    *,
    source_dataset_name: str,
    target_dataset_name: str,
) -> str:
    prefix = f"{source_dataset_name}-"
    if example_id.startswith(prefix):
        return f"{target_dataset_name}-{example_id[len(prefix):]}"
    return example_id


if __name__ == "__main__":
    main()
