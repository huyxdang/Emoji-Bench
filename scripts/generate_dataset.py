#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.benchmark import generate_benchmark_instance
from emoji_bench.formatter import system_to_json
from emoji_bench.generator import generate_system
from emoji_bench.dataset import (
    DEFAULT_TARGET_LENGTHS,
    DEFAULT_VARIANTS,
    DIFFICULTY_CONFIGS,
    DatasetManifest,
    DatasetVariant,
    _error_seed_for_variant,
    _example_record,
    _git_commit,
    _seed_root,
    _select_chain_seed,
    _variant_seed_offsets,
    generate_dataset_records,
    push_dataset_to_hub,
    write_dataset,
)
from emoji_bench.benchmark_types import Condition, ErrorType


def _variant_aliases() -> dict[str, DatasetVariant]:
    aliases: dict[str, DatasetVariant] = {}
    for variant in DEFAULT_VARIANTS:
        aliases[variant.name] = variant
        if variant.error_type is not None:
            aliases[variant.error_type.value.lower().replace("-", "_")] = variant
    aliases["clean"] = DEFAULT_VARIANTS[0]
    return aliases


def _parse_variants(raw_values: list[str] | None) -> tuple[DatasetVariant, ...]:
    if not raw_values:
        return DEFAULT_VARIANTS

    aliases = _variant_aliases()
    selected: list[DatasetVariant] = []
    seen: set[DatasetVariant] = set()

    for raw in raw_values:
        for item in raw.split(","):
            normalized = item.strip().lower().replace("-", "_")
            if not normalized:
                continue
            if normalized == "all":
                return DEFAULT_VARIANTS
            if normalized not in aliases:
                valid = ", ".join(sorted({"all", *aliases.keys()}))
                raise ValueError(f"unknown error type '{item}'. Expected one of: {valid}")
            variant = aliases[normalized]
            if variant not in seen:
                selected.append(variant)
                seen.add(variant)

    if not selected:
        raise ValueError("at least one --error-type must be selected")
    return tuple(selected)


def _manifest_from_records(
    *,
    dataset_name: str,
    bases_per_difficulty: int,
    target_lengths: dict[str, int],
    split_records: dict[str, list[dict]],
    generator_commit: str | None,
) -> DatasetManifest:
    split_counts = {split: len(records) for split, records in split_records.items()}
    difficulty_counts = Counter(
        record["difficulty"]
        for records in split_records.values()
        for record in records
    )
    condition_counts = Counter(
        record["condition"]
        for records in split_records.values()
        for record in records
    )
    error_type_counts = Counter(
        record["error_type"] or "clean"
        for records in split_records.values()
        for record in records
    )

    return DatasetManifest(
        dataset_name=dataset_name,
        total_examples=sum(split_counts.values()),
        bases_per_difficulty=bases_per_difficulty,
        target_lengths=target_lengths,
        split_counts=dict(split_counts),
        difficulty_counts=dict(difficulty_counts),
        condition_counts=dict(condition_counts),
        error_type_counts=dict(error_type_counts),
        generator_commit=generator_commit,
    )


def _trim_records(
    *,
    split_records: dict[str, list[dict]],
    count: int,
) -> dict[str, list[dict]]:
    return {
        "train": [],
        "validation": [],
        "test": list(split_records["test"][:count]),
    }


def _generate_with_count(
    *,
    dataset_name: str,
    bases_per_difficulty: int,
    count: int,
    master_seed: int,
    train_ratio: float,
    validation_ratio: float,
    target_lengths: dict[str, int],
    variants: tuple[DatasetVariant, ...],
) -> tuple[dict[str, list[dict]], DatasetManifest]:
    if train_ratio != 0.0 or validation_ratio != 0.0:
        raise ValueError("--count currently requires --train-ratio 0 --validation-ratio 0")
    if count < 1:
        raise ValueError("--count must be >= 1")

    difficulty_names = tuple(DIFFICULTY_CONFIGS)
    per_difficulty_targets = {
        difficulty: count // len(difficulty_names) + (1 if index < count % len(difficulty_names) else 0)
        for index, difficulty in enumerate(difficulty_names)
    }
    seed_offsets = _variant_seed_offsets(variants)
    split_records: dict[str, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    example_index = 0
    bases_used_per_difficulty: dict[str, int] = {difficulty: 0 for difficulty in difficulty_names}

    for difficulty_index, difficulty_name in enumerate(difficulty_names):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        target_step_count = target_lengths[difficulty_name]
        base_index = 0

        while len([r for r in split_records["test"] if r["difficulty"] == difficulty_name]) < per_difficulty_targets[difficulty_name]:
            seed_root = _seed_root(master_seed, difficulty_index, base_index)
            system_seed = seed_root + 11
            base_id = f"{difficulty_name}-{base_index:04d}"
            system = generate_system(random_seed=system_seed, **config)

            try:
                chain_seed, active_variants = _select_chain_seed(
                    system=system,
                    target_step_count=target_step_count,
                    seed_root=seed_root,
                    variants=variants,
                    seed_offsets=seed_offsets,
                )
            except RuntimeError:
                base_index += 1
                continue

            system_json = system_to_json(system)
            for variant in active_variants:
                difficulty_count = len(
                    [r for r in split_records["test"] if r["difficulty"] == difficulty_name]
                )
                if difficulty_count >= per_difficulty_targets[difficulty_name]:
                    break

                error_seed = _error_seed_for_variant(seed_root, variant, seed_offsets)
                instance = generate_benchmark_instance(
                    system,
                    length=target_step_count,
                    condition=variant.condition,
                    error_type=variant.error_type or ErrorType.E_RES,
                    chain_seed=chain_seed,
                    error_seed=error_seed,
                    instance_id=f"{base_id}-{variant.name}",
                )
                split_records["test"].append(
                    _example_record(
                        dataset_name=dataset_name,
                        base_id=base_id,
                        example_index=example_index,
                        split="test",
                        difficulty=difficulty_name,
                        target_step_count=target_step_count,
                        system_seed=system_seed,
                        chain_seed=chain_seed,
                        error_seed=error_seed,
                        variant=variant,
                        instance=instance,
                        system_json=system_json,
                    )
                )
                example_index += 1

            base_index += 1
            bases_used_per_difficulty[difficulty_name] = base_index

    manifest = _manifest_from_records(
        dataset_name=dataset_name,
        bases_per_difficulty=max(bases_used_per_difficulty.values(), default=0),
        target_lengths=target_lengths,
        split_records=split_records,
        generator_commit=_git_commit(),
    )
    return split_records, manifest


def _parse_length_overrides(raw: str | None) -> dict[str, int]:
    if raw is None:
        return {}

    overrides: dict[str, int] = {}
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                "length overrides must use the format "
                "'easy=3,medium=5,hard=7,expert=10'"
            )
        key = key.strip().lower()
        if key not in DEFAULT_TARGET_LENGTHS:
            raise ValueError(f"unknown difficulty in length overrides: {key}")
        overrides[key] = int(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Emoji-Bench dataset splits and optionally push them to Hugging Face.",
    )
    parser.add_argument(
        "--dataset-name",
        default="emoji-bench-v1",
        help="Dataset name prefix used for example ids and manifests.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/emoji-bench-v1",
        help="Directory where JSONL splits and manifest files will be written.",
    )
    parser.add_argument(
        "--bases-per-difficulty",
        type=int,
        default=50,
        help="Number of base problems to generate for each difficulty level.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260401,
        help="Master seed used to derive system, chain, and error seeds.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of base problems assigned to the train split.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of base problems assigned to the validation split.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Optional shared target derivation length for all difficulties.",
    )
    parser.add_argument(
        "--length-overrides",
        default=None,
        help="Optional per-difficulty target lengths, e.g. easy=3,medium=5,hard=7,expert=10",
    )
    parser.add_argument(
        "--error-type",
        action="append",
        default=None,
        help=(
            "Optional error type filter. Accepts clean, e_res, e_inv, e_casc, "
            "or enum forms like E-CASC. Repeat or comma-separate to select multiple."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Optional exact number of examples to write. Requires train/validation ratios of 0.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id, e.g. huyxdang/emoji-bench-v1.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the generated folder to a Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private when pushing.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN / HUGGINGFACE_HUB_TOKEN.",
    )
    args = parser.parse_args()

    target_lengths = dict(DEFAULT_TARGET_LENGTHS)
    if args.target_length is not None:
        target_lengths = {
            difficulty: args.target_length for difficulty in DEFAULT_TARGET_LENGTHS
        }
    target_lengths.update(_parse_length_overrides(args.length_overrides))
    variants = _parse_variants(args.error_type)

    if args.count is None:
        split_records, manifest = generate_dataset_records(
            dataset_name=args.dataset_name,
            bases_per_difficulty=args.bases_per_difficulty,
            master_seed=args.master_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            target_lengths=target_lengths,
            variants=variants,
        )
    else:
        split_records, manifest = _generate_with_count(
            dataset_name=args.dataset_name,
            bases_per_difficulty=args.bases_per_difficulty,
            count=args.count,
            master_seed=args.master_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            target_lengths=target_lengths,
            variants=variants,
        )
    output_dir = write_dataset(
        args.output_dir,
        split_records,
        manifest,
        repo_id=args.repo_id,
    )

    summary = {
        "output_dir": str(Path(output_dir).resolve()),
        "total_examples": manifest.total_examples,
        "split_counts": manifest.split_counts,
        "difficulty_counts": manifest.difficulty_counts,
        "error_type_counts": manifest.error_type_counts,
        "selected_variants": [
            variant.error_type.value if variant.error_type is not None else "clean"
            for variant in variants
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.push_to_hub:
        if args.repo_id is None:
            raise ValueError("--repo-id is required when using --push-to-hub")
        push_dataset_to_hub(
            output_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
