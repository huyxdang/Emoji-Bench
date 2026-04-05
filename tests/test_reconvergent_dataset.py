import json
import subprocess
import sys
from pathlib import Path

from emoji_bench.reconvergent_dataset import generate_reconvergent_dataset_records


def test_generate_reconvergent_dataset_records_produces_exact_count():
    split_records, manifest = generate_reconvergent_dataset_records(
        dataset_name="emoji-bench-e-reconv-test",
        count=4,
        master_seed=123,
        target_lengths={
            "easy": 4,
            "medium": 4,
            "hard": 4,
            "expert": 4,
        },
    )

    assert split_records["train"] == []
    assert split_records["validation"] == []
    assert len(split_records["test"]) == 4
    assert manifest.total_examples == 4
    assert manifest.split_counts == {"train": 0, "validation": 0, "test": 4}
    assert manifest.error_type_counts == {"E-RECONV": 4}
    assert {record["error_type"] for record in split_records["test"]} == {"E-RECONV"}
    assert all(record["condition"] == "error_injected" for record in split_records["test"])
    assert all(record["expected_error_step"] is not None for record in split_records["test"])


def test_generate_reconvergent_dataset_script_supports_exact_count(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_reconvergent_dataset.py",
            "--dataset-name",
            "emoji-bench-e-reconv-cli",
            "--output-dir",
            str(output_dir),
            "--count",
            "6",
            "--target-length",
            "4",
            "--master-seed",
            "123",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["total_examples"] == 6
    assert summary["error_type_counts"] == {"E-RECONV": 6}
    assert summary["selected_variants"] == ["E-RECONV"]
    assert manifest["total_examples"] == 6
    assert manifest["error_type_counts"] == {"E-RECONV": 6}
