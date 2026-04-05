from __future__ import annotations

import subprocess
from collections import Counter
from typing import Any

from emoji_bench.benchmark import generate_benchmark_instance
from emoji_bench.benchmark_types import Condition, ErrorType
from emoji_bench.dataset import (
    DEFAULT_TARGET_LENGTHS,
    DIFFICULTY_CONFIGS,
    DatasetManifest,
    DatasetVariant,
)
from emoji_bench.formatter import system_to_json
from emoji_bench.generator import generate_system


RECONVERGENT_VARIANT = DatasetVariant(
    name="e_reconv",
    condition=Condition.ERROR_INJECTED,
    error_type=ErrorType.E_RECONV,
    has_error=True,
)
DEFAULT_RECONVERGENT_TARGET_LENGTHS: dict[str, int] = {
    difficulty: max(length, 4)
    for difficulty, length in DEFAULT_TARGET_LENGTHS.items()
}
MAX_CHAIN_SEED_ATTEMPTS = 250
MAX_BASE_ATTEMPTS_PER_DIFFICULTY = 20_000
ERROR_SEED_OFFSET = 101


def _seed_root(master_seed: int, difficulty_index: int, base_index: int) -> int:
    return master_seed * 1_000_000 + difficulty_index * 10_000 + base_index * 100


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _per_difficulty_targets(count: int) -> dict[str, int]:
    difficulty_names = tuple(DIFFICULTY_CONFIGS)
    return {
        difficulty: count // len(difficulty_names) + (1 if index < count % len(difficulty_names) else 0)
        for index, difficulty in enumerate(difficulty_names)
    }


def _select_chain_seed(
    *,
    system: Any,
    target_step_count: int,
    seed_root: int,
) -> tuple[int, int]:
    for attempt in range(MAX_CHAIN_SEED_ATTEMPTS):
        chain_seed = seed_root + 29 + attempt
        error_seed = seed_root + ERROR_SEED_OFFSET
        try:
            generate_benchmark_instance(
                system,
                length=target_step_count,
                condition=Condition.ERROR_INJECTED,
                error_type=ErrorType.E_RECONV,
                chain_seed=chain_seed,
                error_seed=error_seed,
            )
        except ValueError:
            continue
        return chain_seed, error_seed

    raise RuntimeError(
        "Failed to find a reconvergent derivation for the generated system "
        f"after {MAX_CHAIN_SEED_ATTEMPTS} chain attempts"
    )


def _example_record(
    *,
    dataset_name: str,
    base_id: str,
    example_index: int,
    difficulty: str,
    target_step_count: int,
    system_seed: int,
    chain_seed: int,
    error_seed: int,
    instance: Any,
    system_json: str,
) -> dict[str, Any]:
    assert instance.error_info is not None
    return {
        "example_id": f"{dataset_name}-{example_index:06d}",
        "base_id": base_id,
        "split": "test",
        "difficulty": difficulty,
        "condition": RECONVERGENT_VARIANT.condition.value,
        "error_type": ErrorType.E_RECONV.value,
        "has_error": True,
        "prompt": instance.prompt,
        "actual_step_count": len(instance.chain.steps),
        "target_step_count": target_step_count,
        "expected_error_step": instance.error_info.step_number,
        "system_json": system_json,
        "system_seed": system_seed,
        "chain_seed": chain_seed,
        "error_seed": error_seed,
    }


def generate_reconvergent_dataset_records(
    *,
    dataset_name: str,
    count: int,
    master_seed: int,
    target_lengths: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], DatasetManifest]:
    if count < 1:
        raise ValueError("count must be >= 1")

    resolved_lengths = dict(DEFAULT_RECONVERGENT_TARGET_LENGTHS)
    if target_lengths is not None:
        resolved_lengths.update(target_lengths)

    targets = _per_difficulty_targets(count)
    produced_per_difficulty = {difficulty: 0 for difficulty in DIFFICULTY_CONFIGS}
    bases_used_per_difficulty = {difficulty: 0 for difficulty in DIFFICULTY_CONFIGS}
    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    example_index = 0

    for difficulty_index, difficulty_name in enumerate(DIFFICULTY_CONFIGS):
        config = DIFFICULTY_CONFIGS[difficulty_name]
        target_step_count = resolved_lengths[difficulty_name]
        base_index = 0

        while produced_per_difficulty[difficulty_name] < targets[difficulty_name]:
            if base_index >= MAX_BASE_ATTEMPTS_PER_DIFFICULTY:
                raise RuntimeError(
                    "Failed to generate enough reconvergent bases for "
                    f"{difficulty_name} after {MAX_BASE_ATTEMPTS_PER_DIFFICULTY} attempts"
                )

            seed_root = _seed_root(master_seed, difficulty_index, base_index)
            system_seed = seed_root + 11
            base_id = f"{difficulty_name}-{base_index:04d}"
            system = generate_system(random_seed=system_seed, **config)

            try:
                chain_seed, error_seed = _select_chain_seed(
                    system=system,
                    target_step_count=target_step_count,
                    seed_root=seed_root,
                )
            except RuntimeError:
                base_index += 1
                continue

            instance = generate_benchmark_instance(
                system,
                length=target_step_count,
                condition=Condition.ERROR_INJECTED,
                error_type=ErrorType.E_RECONV,
                chain_seed=chain_seed,
                error_seed=error_seed,
                instance_id=f"{base_id}-{RECONVERGENT_VARIANT.name}",
            )
            split_records["test"].append(
                _example_record(
                    dataset_name=dataset_name,
                    base_id=base_id,
                    example_index=example_index,
                    difficulty=difficulty_name,
                    target_step_count=target_step_count,
                    system_seed=system_seed,
                    chain_seed=chain_seed,
                    error_seed=error_seed,
                    instance=instance,
                    system_json=system_to_json(system),
                )
            )
            example_index += 1
            produced_per_difficulty[difficulty_name] += 1
            base_index += 1
            bases_used_per_difficulty[difficulty_name] = base_index

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
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        total_examples=sum(split_counts.values()),
        bases_per_difficulty=max(bases_used_per_difficulty.values(), default=0),
        target_lengths=resolved_lengths,
        split_counts=dict(split_counts),
        difficulty_counts=dict(difficulty_counts),
        condition_counts=dict(condition_counts),
        error_type_counts=dict(error_type_counts),
        generator_commit=_git_commit(),
    )
    return split_records, manifest
