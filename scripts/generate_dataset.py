#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emoji_bench.dataset import (
    DEFAULT_TARGET_LENGTHS,
    generate_dataset_records,
    push_dataset_to_hub,
    write_dataset,
)


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

    split_records, manifest = generate_dataset_records(
        dataset_name=args.dataset_name,
        bases_per_difficulty=args.bases_per_difficulty,
        master_seed=args.master_seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        target_lengths=target_lengths,
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
