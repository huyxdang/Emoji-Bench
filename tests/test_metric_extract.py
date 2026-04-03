import csv
import json

from emoji_bench.metric_extract import (
    build_key_metric_payload,
    build_long_rows,
    build_wide_rows,
    load_report_summary,
    write_key_metric_artifacts,
)


def test_extract_key_metrics_from_report_summary(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-03T00:00:00+00:00",
                "overall": {
                    "detection_accuracy": 0.75,
                    "joint_accuracy": 0.5,
                    "false_positive": 2,
                },
                "by_model": [
                    {
                        "model": "model-a",
                        "detection_accuracy": 0.8,
                        "joint_accuracy": 0.6,
                        "false_positive": 1,
                    },
                ],
                "by_model_difficulty": [
                    {
                        "model": "model-a",
                        "difficulty": "easy",
                        "detection_accuracy": 1.0,
                        "joint_accuracy": 0.75,
                        "false_positive": 0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report, loaded_path = load_report_summary(summary_path)
    payload = build_key_metric_payload(report, source_summary_path=loaded_path)

    assert loaded_path == summary_path
    assert payload["overall"] == {
        "scope": "overall",
        "model": None,
        "difficulty": None,
        "detection_rate": 0.75,
        "joint_detection": 0.5,
        "false_positives": 2,
    }
    assert payload["by_model"] == [
        {
            "scope": "model",
            "model": "model-a",
            "difficulty": None,
            "detection_rate": 0.8,
            "joint_detection": 0.6,
            "false_positives": 1,
        },
    ]
    assert payload["by_model_difficulty"] == [
        {
            "scope": "model_difficulty",
            "model": "model-a",
            "difficulty": "easy",
            "detection_rate": 1.0,
            "joint_detection": 0.75,
            "false_positives": 0,
        },
    ]

    wide_rows = build_wide_rows(payload)
    long_rows = build_long_rows(payload)
    assert len(wide_rows) == 3
    assert len(long_rows) == 9

    output_dir = write_key_metric_artifacts(payload, output_dir=tmp_path / "key-metrics")
    assert (output_dir / "key_metrics.json").exists()
    assert (output_dir / "key_metrics.csv").exists()
    assert (output_dir / "key_metrics_overall.csv").exists()
    assert (output_dir / "key_metrics_by_model.csv").exists()
    assert (output_dir / "key_metrics_by_model_difficulty.csv").exists()
    assert (output_dir / "key_metrics_long.csv").exists()

    csv_rows = list(csv.DictReader((output_dir / "key_metrics.csv").open(encoding="utf-8")))
    assert csv_rows == [
        {
            "scope": "overall",
            "model": "",
            "difficulty": "",
            "detection_rate": "0.75",
            "joint_detection": "0.5",
            "false_positives": "2",
        },
        {
            "scope": "model",
            "model": "model-a",
            "difficulty": "",
            "detection_rate": "0.8",
            "joint_detection": "0.6",
            "false_positives": "1",
        },
        {
            "scope": "model_difficulty",
            "model": "model-a",
            "difficulty": "easy",
            "detection_rate": "1.0",
            "joint_detection": "0.75",
            "false_positives": "0",
        },
    ]
