import json

from emoji_bench.reporting import (
    load_prediction_rows,
    summarize_prediction_rows,
    write_report_artifacts,
)


def test_summarize_prediction_rows_reports_detection_and_localization_metrics():
    rows = [
        {
            "example_id": "ex-1",
            "model": "model-a",
            "provider": "openai",
            "difficulty": "easy",
            "error_type": None,
            "actual_step_count": 3,
            "expected_has_error": False,
            "expected_error_step": None,
            "predicted_has_error": False,
            "predicted_error_step": None,
            "request_latency_seconds": 1.2,
            "total_tokens": 30,
        },
        {
            "example_id": "ex-2",
            "model": "model-a",
            "provider": "openai",
            "difficulty": "medium",
            "error_type": "E-RES",
            "actual_step_count": 5,
            "expected_has_error": True,
            "expected_error_step": 3,
            "predicted_has_error": True,
            "predicted_error_step": 3,
            "request_latency_seconds": 1.8,
            "total_tokens": 35,
        },
        {
            "example_id": "ex-3",
            "model": "model-b",
            "provider": "anthropic",
            "difficulty": "hard",
            "error_type": "E-CASC",
            "actual_step_count": 7,
            "expected_has_error": True,
            "expected_error_step": 2,
            "predicted_has_error": True,
            "predicted_error_step": 4,
            "request_latency_seconds": 2.6,
            "total_tokens": 42,
        },
        {
            "example_id": "ex-4",
            "model": "model-b",
            "provider": "anthropic",
            "difficulty": "expert",
            "error_type": None,
            "actual_step_count": 10,
            "expected_has_error": False,
            "expected_error_step": None,
            "predicted_has_error": True,
            "predicted_error_step": 1,
            "request_latency_seconds": 3.1,
            "total_tokens": 50,
        },
    ]

    report = summarize_prediction_rows(rows)

    overall = report["overall"]
    assert report["total_rows"] == 4
    assert report["models"] == ["model-a", "model-b"]
    assert overall["detection_accuracy"] == 0.75
    assert round(overall["detection_precision"], 6) == round(2 / 3, 6)
    assert overall["detection_recall"] == 1.0
    assert overall["detection_f1"] == 0.8
    assert overall["joint_accuracy"] == 0.5
    assert overall["localization_accuracy_on_error_rows"] == 0.5
    assert overall["localization_accuracy_when_detected"] == 0.5
    assert overall["false_positive_rate_clean"] == 0.5
    assert overall["false_negative_rate_error"] == 0.0
    assert overall["mean_abs_step_error"] == 1.0
    assert overall["median_abs_step_error"] == 1.0
    assert overall["within_one_step_rate"] == 0.5
    assert overall["off_by_one_rate"] == 0.0
    assert overall["latency_mean_seconds"] == 2.175
    assert overall["total_tokens_mean"] == 39.25

    by_model = {row["model"]: row for row in report["by_model"]}
    assert by_model["model-a"]["joint_accuracy"] == 1.0
    assert by_model["model-b"]["joint_accuracy"] == 0.0


def test_load_prediction_rows_and_write_report_artifacts(tmp_path):
    eval_dir = tmp_path / "artifacts" / "evals" / "run-a"
    eval_dir.mkdir(parents=True)
    predictions_path = eval_dir / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "example_id": "ex-1",
                "model": "model-a",
                "provider": "openai",
                "difficulty": "easy",
                "error_type": None,
                "actual_step_count": 3,
                "expected_has_error": False,
                "expected_error_step": None,
                "predicted_has_error": False,
                "predicted_error_step": None,
                "request_latency_seconds": 1.0,
                "total_tokens": 20,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows, input_paths = load_prediction_rows([tmp_path / "artifacts"])
    assert len(rows) == 1
    assert input_paths == [str(predictions_path)]
    assert rows[0]["_source_path"] == str(predictions_path)

    report = summarize_prediction_rows(rows)
    output_dir = write_report_artifacts(
        report,
        output_dir=tmp_path / "report",
        input_paths=input_paths,
    )

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "by_model.csv").exists()
    assert (output_dir / "report.html").exists()
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["input_paths"] == input_paths
