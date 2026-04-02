import json
import sys

from emoji_bench import eval_cli
from emoji_bench.provider_eval import ProviderResponse


def test_select_shard_records_partitions_examples_by_stable_hash():
    records = [{"example_id": f"ex-{idx}"} for idx in range(12)]

    shard_groups = [
        eval_cli._select_shard_records(records, shard_index=shard_index, num_shards=4)
        for shard_index in range(4)
    ]

    flattened_ids = [record["example_id"] for group in shard_groups for record in group]
    assert sorted(flattened_ids) == [record["example_id"] for record in records]
    assert len(flattened_ids) == len(set(flattened_ids))
    for shard_index, group in enumerate(shard_groups):
        assert all(
            eval_cli._stable_shard_index(record["example_id"], num_shards=4) == shard_index
            for record in group
        )


def test_resolve_output_dir_appends_shard_directory_for_sharded_runs(tmp_path):
    input_path = tmp_path / "demo-dataset" / "test.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("", encoding="utf-8")

    output_dir = eval_cli._resolve_output_dir(
        raw_output_dir=None,
        input_path=input_path,
        model_key="gpt-5.4-mini",
        shard_index=3,
        num_shards=12,
    )

    assert output_dir == (
        eval_cli._default_output_dir(input_path, "gpt-5.4-mini") / "shard-03-of-12"
    )


def test_main_writes_predictions_and_summary_into_shard_subdirectory(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "demo-dataset"
    dataset_dir.mkdir()
    input_path = dataset_dir / "test.jsonl"
    records = [
        {
            "example_id": f"ex-{idx}",
            "prompt": f"prompt-{idx}",
            "difficulty": "easy",
            "error_type": None,
            "has_error": idx % 2 == 0,
            "expected_error_step": 2 if idx % 2 == 0 else None,
        }
        for idx in range(6)
    ]
    input_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    prompt_to_record = {record["prompt"]: record for record in records}

    monkeypatch.setattr(eval_cli, "resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(eval_cli, "make_client", lambda provider, *, api_key: object())

    def fake_request_prediction(*, prompt, **kwargs):
        record = prompt_to_record[prompt]
        return ProviderResponse(
            prediction_payload={
                "has_error": record["has_error"],
                "error_step": record["expected_error_step"],
            },
            response_id=f"response-{record['example_id']}",
            raw_output_text="{}",
        )

    monkeypatch.setattr(eval_cli, "request_prediction", fake_request_prediction)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/evaluate_model.py",
            str(dataset_dir),
            "--model",
            "gpt-5.4-mini",
            "--num-shards",
            "2",
            "--shard-index",
            "1",
            "--output-dir",
            str(tmp_path / "eval-output"),
        ],
    )

    eval_cli.main(
        description="test evaluator",
        default_model="gpt-5.4-mini",
    )

    shard_records = eval_cli._select_shard_records(records, shard_index=1, num_shards=2)
    output_dir = tmp_path / "eval-output" / "shard-01-of-02"
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    prediction_rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert output_dir.exists()
    assert [row["example_id"] for row in prediction_rows] == [
        record["example_id"] for record in shard_records
    ]
    assert {row["shard_label"] for row in prediction_rows} == {"shard-01-of-02"}
    assert summary["num_shards"] == 2
    assert summary["shard_index"] == 1
    assert summary["shard_label"] == "shard-01-of-02"
    assert summary["shard_total_examples"] == len(shard_records)
