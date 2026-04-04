import json
import subprocess
import sys
from pathlib import Path

from emoji_bench.dataset import generate_dataset_records, write_dataset


def test_relabel_dataset_numeric_script_rewrites_prompts_and_systems(tmp_path):
    split_records, manifest = generate_dataset_records(
        dataset_name="emoji-bench-test",
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
    input_dir = write_dataset(tmp_path / "input", split_records, manifest)
    output_dir = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/relabel_dataset_numeric.py",
            str(input_dir),
            str(output_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    manifest_data = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["dataset_name"] == "emoji-bench-test-numbers"
    assert manifest_data["total_examples"] == manifest.total_examples

    original_record = json.loads((input_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    first_record = json.loads((output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_record["example_id"].startswith("emoji-bench-test-numbers-")
    assert first_record["prompt"] != original_record["prompt"]
    original_system = json.loads(original_record["system_json"])
    for symbol in original_system["symbols"]:
        assert symbol not in first_record["prompt"]
        assert symbol not in first_record["system_json"]
    relabeled_system = json.loads(first_record["system_json"])
    assert all(symbol.isdigit() and len(symbol) == 2 for symbol in relabeled_system["symbols"])
