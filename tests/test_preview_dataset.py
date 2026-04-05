import json
import subprocess
import sys
from pathlib import Path

from emoji_bench.dataset import generate_dataset_records, write_dataset


def test_preview_dataset_script_renders_metadata_and_prompt(tmp_path):
    split_records, manifest = generate_dataset_records(
        dataset_name="emoji-bench-preview-test",
        bases_per_difficulty=1,
        master_seed=123,
        train_ratio=0.0,
        validation_ratio=0.0,
        target_lengths={
            "easy": 3,
            "medium": 3,
            "hard": 3,
            "expert": 3,
        },
    )
    output_dir = write_dataset(tmp_path / "dataset", split_records, manifest)
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_dataset.py",
            str(output_dir),
            "--count",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    first_record = json.loads((output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "=== DATASET ===" in result.stdout
    assert "Example 1/1" in result.stdout
    assert f"example_id: {first_record['example_id']}" in result.stdout
    assert first_record["prompt"] in result.stdout


def test_preview_dataset_script_can_select_specific_example(tmp_path):
    split_records, manifest = generate_dataset_records(
        dataset_name="emoji-bench-preview-test",
        bases_per_difficulty=1,
        master_seed=123,
        train_ratio=0.0,
        validation_ratio=0.0,
        target_lengths={
            "easy": 3,
            "medium": 3,
            "hard": 3,
            "expert": 3,
        },
    )
    output_dir = write_dataset(tmp_path / "dataset", split_records, manifest)
    repo_root = Path(__file__).resolve().parents[1]
    records = [json.loads(line) for line in (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    target = records[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_dataset.py",
            str(output_dir / "test.jsonl"),
            "--example-id",
            target["example_id"],
            "--no-manifest",
            "--prompt-only",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "=== DATASET ===" not in result.stdout
    assert "example_id:" not in result.stdout
    assert target["prompt"] in result.stdout
