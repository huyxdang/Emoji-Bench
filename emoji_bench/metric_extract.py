from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("detection_accuracy", "Detection Rate", "detection_rate"),
    ("joint_accuracy", "Joint Detection", "joint_detection"),
    ("false_positive", "False Positives", "false_positives"),
)

WIDE_FIELDNAMES = [
    "scope",
    "model",
    "difficulty",
    "detection_rate",
    "joint_detection",
    "false_positives",
]

LONG_FIELDNAMES = [
    "scope",
    "model",
    "difficulty",
    "metric",
    "metric_label",
    "value",
]


def resolve_summary_path(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir)
    if path.exists():
        summary_path = path / "summary.json" if path.is_dir() else path
    else:
        summary_path = path if path.suffix == ".json" else path / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json found at: {summary_path}")

    return summary_path


def load_report_summary(path_or_dir: str | Path) -> tuple[dict[str, Any], Path]:
    summary_path = resolve_summary_path(path_or_dir)
    report = json.loads(summary_path.read_text(encoding="utf-8"))

    required_keys = {"overall", "by_model", "by_model_difficulty"}
    missing = sorted(required_keys - set(report))
    if missing:
        raise KeyError(
            f"Report summary is missing required keys: {', '.join(missing)}",
        )

    return report, summary_path


def _project_row(
    row: dict[str, Any],
    *,
    scope: str,
    model: str | None = None,
    difficulty: str | None = None,
) -> dict[str, Any]:
    projected = {
        "scope": scope,
        "model": model,
        "difficulty": difficulty,
    }
    for source_key, _label, target_key in METRIC_FIELDS:
        if source_key not in row:
            raise KeyError(
                f"Row is missing required metric '{source_key}' for scope '{scope}'",
            )
        projected[target_key] = row[source_key]
    return projected


def build_key_metric_payload(
    report: dict[str, Any],
    *,
    source_summary_path: str | Path,
) -> dict[str, Any]:
    overall = _project_row(report["overall"], scope="overall")
    by_model = [
        _project_row(row, scope="model", model=row["model"])
        for row in report.get("by_model", [])
    ]
    by_model_difficulty = [
        _project_row(
            row,
            scope="model_difficulty",
            model=row["model"],
            difficulty=row["difficulty"],
        )
        for row in report.get("by_model_difficulty", [])
    ]

    return {
        "generated_at": report.get("generated_at"),
        "source_summary_path": str(Path(source_summary_path)),
        "metric_definitions": [
            {
                "source_field": source_key,
                "label": label,
                "key": target_key,
            }
            for source_key, label, target_key in METRIC_FIELDS
        ],
        "overall": overall,
        "by_model": by_model,
        "by_model_difficulty": by_model_difficulty,
    }


def build_wide_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        payload["overall"],
        *payload["by_model"],
        *payload["by_model_difficulty"],
    ]


def build_long_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wide_row in build_wide_rows(payload):
        for _source_key, label, target_key in METRIC_FIELDS:
            rows.append(
                {
                    "scope": wide_row["scope"],
                    "model": wide_row["model"],
                    "difficulty": wide_row["difficulty"],
                    "metric": target_key,
                    "metric_label": label,
                    "value": wide_row[target_key],
                }
            )
    return rows


def _serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _serialize_csv_value(row.get(field)) for field in fieldnames},
            )


def write_key_metric_artifacts(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wide_rows = build_wide_rows(payload)
    long_rows = build_long_rows(payload)

    (output_path / "key_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_path / "key_metrics.csv", wide_rows, WIDE_FIELDNAMES)
    _write_csv(output_path / "key_metrics_overall.csv", [payload["overall"]], WIDE_FIELDNAMES)
    _write_csv(output_path / "key_metrics_by_model.csv", payload["by_model"], WIDE_FIELDNAMES)
    _write_csv(
        output_path / "key_metrics_by_model_difficulty.csv",
        payload["by_model_difficulty"],
        WIDE_FIELDNAMES,
    )
    _write_csv(output_path / "key_metrics_long.csv", long_rows, LONG_FIELDNAMES)

    return output_path
