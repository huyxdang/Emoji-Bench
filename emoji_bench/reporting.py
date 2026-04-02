from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

from emoji_bench.evaluation import load_jsonl_records


DIFFICULTY_ORDER: dict[str, int] = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "expert": 3,
}

ERROR_TYPE_ORDER: dict[str, int] = {
    "clean": 0,
    "E-RES": 1,
    "E-INV": 2,
    "E-CASC": 3,
}


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: Iterable[int | float]) -> float | None:
    seq = list(values)
    if not seq:
        return None
    return sum(seq) / len(seq)


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    seq = sorted(values)
    if not seq:
        return None
    if len(seq) == 1:
        return seq[0]

    rank = percentile * (len(seq) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return seq[lower_index]

    lower_value = seq[lower_index]
    upper_value = seq[upper_index]
    fraction = rank - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"Expected boolean-like value, got {value!r}")


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    raise ValueError(f"Expected int-like or null value, got {value!r}")


def _as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)
    raise ValueError(f"Expected float-like or null value, got {value!r}")


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    expected_has_error = _as_bool(row["expected_has_error"])
    predicted_has_error = _as_bool(row["predicted_has_error"])
    expected_error_step = _as_int_or_none(row.get("expected_error_step"))
    predicted_error_step = _as_int_or_none(row.get("predicted_error_step"))
    if not predicted_has_error:
        predicted_error_step = None

    has_error_correct = row.get("has_error_correct")
    if has_error_correct is None:
        has_error_correct = predicted_has_error == expected_has_error
    else:
        has_error_correct = _as_bool(has_error_correct)

    error_step_correct = row.get("error_step_correct")
    if error_step_correct is None:
        error_step_correct = predicted_error_step == expected_error_step
    else:
        error_step_correct = _as_bool(error_step_correct)

    joint_correct = row.get("joint_correct")
    if joint_correct is None:
        joint_correct = has_error_correct and error_step_correct
    else:
        joint_correct = _as_bool(joint_correct)

    return {
        "example_id": row["example_id"],
        "model": row.get("model", "unknown"),
        "provider": row.get("provider"),
        "difficulty": row["difficulty"],
        "error_type": row.get("error_type") or "clean",
        "actual_step_count": _as_int_or_none(row.get("actual_step_count")),
        "expected_has_error": expected_has_error,
        "expected_error_step": expected_error_step,
        "predicted_has_error": predicted_has_error,
        "predicted_error_step": predicted_error_step,
        "has_error_correct": has_error_correct,
        "error_step_correct": error_step_correct,
        "joint_correct": joint_correct,
        "request_latency_seconds": _as_float_or_none(row.get("request_latency_seconds")),
        "input_tokens": _as_int_or_none(row.get("input_tokens")),
        "output_tokens": _as_int_or_none(row.get("output_tokens")),
        "reasoning_tokens": _as_int_or_none(row.get("reasoning_tokens")),
        "total_tokens": _as_int_or_none(row.get("total_tokens")),
        "source_path": row.get("_source_path"),
    }


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(rows)
    error_rows = [row for row in rows if row["expected_has_error"]]
    clean_rows = [row for row in rows if not row["expected_has_error"]]

    tp = sum(row["expected_has_error"] and row["predicted_has_error"] for row in rows)
    tn = sum((not row["expected_has_error"]) and (not row["predicted_has_error"]) for row in rows)
    fp = sum((not row["expected_has_error"]) and row["predicted_has_error"] for row in rows)
    fn = sum(row["expected_has_error"] and (not row["predicted_has_error"]) for row in rows)

    predicted_step_rows = [
        row
        for row in error_rows
        if row["predicted_error_step"] is not None
    ]
    abs_step_errors = [
        abs(row["predicted_error_step"] - row["expected_error_step"])
        for row in predicted_step_rows
        if row["expected_error_step"] is not None
    ]

    latency_values = [
        row["request_latency_seconds"]
        for row in rows
        if row["request_latency_seconds"] is not None
    ]
    input_tokens = [row["input_tokens"] for row in rows if row["input_tokens"] is not None]
    output_tokens = [row["output_tokens"] for row in rows if row["output_tokens"] is not None]
    reasoning_tokens = [
        row["reasoning_tokens"]
        for row in rows
        if row["reasoning_tokens"] is not None
    ]
    total_tokens = [row["total_tokens"] for row in rows if row["total_tokens"] is not None]

    return {
        "total_rows": total_rows,
        "error_rows": len(error_rows),
        "clean_rows": len(clean_rows),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_error_rows": sum(row["predicted_has_error"] for row in rows),
        "detection_accuracy": _safe_div(tp + tn, total_rows),
        "detection_precision": _safe_div(tp, tp + fp),
        "detection_recall": _safe_div(tp, tp + fn),
        "detection_f1": (
            None
            if _safe_div(tp, tp + fp) is None or _safe_div(tp, tp + fn) is None
            or ((_safe_div(tp, tp + fp) or 0.0) + (_safe_div(tp, tp + fn) or 0.0)) == 0
            else (
                2 * (_safe_div(tp, tp + fp) or 0.0) * (_safe_div(tp, tp + fn) or 0.0)
                / ((_safe_div(tp, tp + fp) or 0.0) + (_safe_div(tp, tp + fn) or 0.0))
            )
        ),
        "false_positive_rate_clean": _safe_div(fp, len(clean_rows)),
        "false_negative_rate_error": _safe_div(fn, len(error_rows)),
        "joint_accuracy": _safe_div(sum(row["joint_correct"] for row in rows), total_rows),
        "error_step_accuracy_all_rows": _safe_div(
            sum(row["error_step_correct"] for row in rows),
            total_rows,
        ),
        "localization_accuracy_on_error_rows": _safe_div(
            sum(row["error_step_correct"] for row in error_rows),
            len(error_rows),
        ),
        "localization_accuracy_when_detected": _safe_div(
            sum(
                row["predicted_error_step"] == row["expected_error_step"]
                for row in error_rows
                if row["predicted_has_error"]
            ),
            sum(row["predicted_has_error"] for row in error_rows),
        ),
        "mean_abs_step_error": _mean(abs_step_errors),
        "median_abs_step_error": _percentile([float(value) for value in abs_step_errors], 0.5),
        "within_one_step_rate": _safe_div(
            sum(error <= 1 for error in abs_step_errors),
            len(abs_step_errors),
        ),
        "off_by_one_rate": _safe_div(
            sum(error == 1 for error in abs_step_errors),
            len(abs_step_errors),
        ),
        "latency_mean_seconds": _mean(latency_values),
        "latency_p50_seconds": _percentile(latency_values, 0.5),
        "latency_p90_seconds": _percentile(latency_values, 0.9),
        "input_tokens_mean": _mean(input_tokens),
        "output_tokens_mean": _mean(output_tokens),
        "reasoning_tokens_mean": _mean(reasoning_tokens),
        "total_tokens_mean": _mean(total_tokens),
    }


def _sorted_group_rows(rows: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        result: list[Any] = []
        for key in keys:
            value = row[key]
            if key == "difficulty":
                result.append(DIFFICULTY_ORDER.get(value, 999))
            elif key == "error_type":
                result.append(ERROR_TYPE_ORDER.get(value, 999))
            elif key in {"expected_error_step", "actual_step_count"}:
                result.append(-1 if value is None else value)
            else:
                result.append(value)
        return tuple(result)

    return sorted(rows, key=sort_key)


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_rows = [_normalize_row(row) for row in rows]

    by_model_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_difficulty_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model_error_type_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model_expected_step_groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_model_step_count_groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in normalized_rows:
        by_model_groups[row["model"]].append(row)
        by_model_difficulty_groups[(row["model"], row["difficulty"])].append(row)
        by_model_error_type_groups[(row["model"], row["error_type"])].append(row)
        if row["expected_has_error"] and row["expected_error_step"] is not None:
            by_model_expected_step_groups[(row["model"], row["expected_error_step"])].append(row)
        if row["actual_step_count"] is not None:
            by_model_step_count_groups[(row["model"], row["actual_step_count"])].append(row)

    by_model = _sorted_group_rows(
        [
            {"model": model, **_compute_metrics(group_rows)}
            for model, group_rows in by_model_groups.items()
        ],
        keys=("model",),
    )
    by_model_difficulty = _sorted_group_rows(
        [
            {"model": model, "difficulty": difficulty, **_compute_metrics(group_rows)}
            for (model, difficulty), group_rows in by_model_difficulty_groups.items()
        ],
        keys=("model", "difficulty"),
    )
    by_model_error_type = _sorted_group_rows(
        [
            {"model": model, "error_type": error_type, **_compute_metrics(group_rows)}
            for (model, error_type), group_rows in by_model_error_type_groups.items()
        ],
        keys=("model", "error_type"),
    )
    by_model_expected_step = _sorted_group_rows(
        [
            {
                "model": model,
                "expected_error_step": expected_error_step,
                **_compute_metrics(group_rows),
            }
            for (model, expected_error_step), group_rows in by_model_expected_step_groups.items()
        ],
        keys=("model", "expected_error_step"),
    )
    by_model_actual_step_count = _sorted_group_rows(
        [
            {
                "model": model,
                "actual_step_count": actual_step_count,
                **_compute_metrics(group_rows),
            }
            for (model, actual_step_count), group_rows in by_model_step_count_groups.items()
        ],
        keys=("model", "actual_step_count"),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(normalized_rows),
        "models": sorted(by_model_groups),
        "overall": _compute_metrics(normalized_rows),
        "by_model": by_model,
        "by_model_difficulty": by_model_difficulty,
        "by_model_error_type": by_model_error_type,
        "by_model_expected_step": by_model_expected_step,
        "by_model_actual_step_count": by_model_actual_step_count,
    }


def load_prediction_rows(paths: Iterable[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    files = resolve_prediction_files(paths)
    rows: list[dict[str, Any]] = []

    for path in files:
        for row in load_jsonl_records(path):
            annotated = dict(row)
            annotated["_source_path"] = str(path)
            rows.append(annotated)

    return rows, [str(path) for path in files]


def resolve_prediction_files(paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            candidate = path
            if candidate not in seen:
                resolved.append(candidate)
                seen.add(candidate)
            continue

        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: {path}")

        direct = path / "predictions.jsonl"
        if direct.exists():
            if direct not in seen:
                resolved.append(direct)
                seen.add(direct)
            continue

        for candidate in sorted(path.rglob("predictions.jsonl")):
            if candidate not in seen:
                resolved.append(candidate)
                seen.add(candidate)

    if not resolved:
        joined = ", ".join(str(Path(p)) for p in paths)
        raise FileNotFoundError(f"No predictions.jsonl files found in: {joined}")

    return resolved


def _serialize_metric_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_report_artifacts(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
    input_paths: list[str],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_payload = dict(report)
    summary_payload["input_paths"] = input_paths
    (output_path / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for name in (
        "by_model",
        "by_model_difficulty",
        "by_model_error_type",
        "by_model_expected_step",
        "by_model_actual_step_count",
    ):
        rows = report[name]
        if not rows:
            continue
        with (output_path / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _serialize_metric_value(value) for key, value in row.items()})

    (output_path / "report.html").write_text(
        render_html_report(report, input_paths=input_paths),
        encoding="utf-8",
    )
    return output_path


def _format_metric(value: float | int | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value * 100:.1f}%"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")
    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
    )


def _metric_color(value: float | None) -> str:
    if value is None:
        return "#f2f2f2"
    bounded = max(0.0, min(1.0, value))
    red = int(245 - 110 * bounded)
    green = int(228 - 58 * bounded)
    blue = int(230 - 140 * bounded)
    return f"rgb({red}, {green}, {blue})"


def _render_bar_chart(
    title: str,
    rows: list[dict[str, Any]],
    *,
    metrics: list[tuple[str, str]],
    label_key: str = "model",
) -> str:
    if not rows:
        return ""

    width = 900
    height = 320
    margin_left = 70
    margin_right = 20
    margin_top = 40
    margin_bottom = 70
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    group_width = chart_width / max(len(rows), 1)
    bar_width = min(28, group_width / max(len(metrics), 1) - 6)

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
    ]
    svg_parts.append(f'<text x="{margin_left}" y="24" class="chart-title">{escape(title)}</text>')
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" '
        f'x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" class="axis" />'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + chart_height}" class="axis" />'
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for tick in range(0, 6):
        value = tick / 5
        y = margin_top + chart_height - value * chart_height
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" class="grid" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" class="axis-label">{value:.1f}</text>'
        )

    for row_index, row in enumerate(rows):
        group_left = margin_left + row_index * group_width + (group_width - bar_width * len(metrics)) / 2
        for metric_index, (metric_key, metric_label) in enumerate(metrics):
            value = row.get(metric_key)
            if value is None:
                continue
            bar_height = max(0.0, min(1.0, float(value))) * chart_height
            x = group_left + metric_index * bar_width
            y = margin_top + chart_height - bar_height
            color = palette[metric_index % len(palette)]
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}">'
                f'<title>{escape(str(row[label_key]))}: {escape(metric_label)} = {value:.3f}</title>'
                "</rect>"
            )

        label_x = margin_left + row_index * group_width + group_width / 2
        svg_parts.append(
            f'<text x="{label_x:.1f}" y="{margin_top + chart_height + 18}" '
            f'text-anchor="middle" class="axis-label">{escape(str(row[label_key]))}</text>'
        )

    legend_y = height - 18
    for metric_index, (_, metric_label) in enumerate(metrics):
        legend_x = margin_left + metric_index * 180
        color = palette[metric_index % len(palette)]
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />'
        )
        svg_parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y}" class="legend-label">{escape(metric_label)}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_heatmap(
    title: str,
    rows: list[dict[str, Any]],
    *,
    row_key: str,
    column_key: str,
    metric_key: str,
    row_order: list[Any] | None = None,
    column_order: list[Any] | None = None,
) -> str:
    if not rows:
        return ""

    row_values = row_order or sorted({row[row_key] for row in rows})
    column_values = column_order or sorted({row[column_key] for row in rows})
    cell_map = {
        (row[row_key], row[column_key]): row.get(metric_key)
        for row in rows
    }

    cell_size = 72
    width = 140 + cell_size * len(column_values)
    height = 90 + cell_size * len(row_values)
    margin_left = 110
    margin_top = 40

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{margin_left}" y="24" class="chart-title">{escape(title)}</text>')

    for column_index, column_value in enumerate(column_values):
        x = margin_left + column_index * cell_size + cell_size / 2
        parts.append(
            f'<text x="{x}" y="{margin_top - 10}" text-anchor="middle" class="axis-label">'
            f"{escape(str(column_value))}</text>"
        )

    for row_index, row_value in enumerate(row_values):
        y = margin_top + row_index * cell_size + cell_size / 2
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" class="axis-label">'
            f"{escape(str(row_value))}</text>"
        )
        for column_index, column_value in enumerate(column_values):
            metric_value = cell_map.get((row_value, column_value))
            x = margin_left + column_index * cell_size
            y_rect = margin_top + row_index * cell_size
            color = _metric_color(metric_value if isinstance(metric_value, float) else None)
            parts.append(
                f'<rect x="{x}" y="{y_rect}" width="{cell_size - 4}" height="{cell_size - 4}" '
                f'rx="8" ry="8" fill="{color}" stroke="#d6d6d6">'
                f'<title>{escape(str(row_value))} / {escape(str(column_value))}: '
                f'{metric_key} = {metric_value if metric_value is not None else "n/a"}</title>'
                "</rect>"
            )
            label = "n/a" if metric_value is None else f"{metric_value * 100:.0f}%"
            parts.append(
                f'<text x="{x + (cell_size - 4) / 2}" y="{y_rect + cell_size / 2}" '
                f'text-anchor="middle" dominant-baseline="middle" class="cell-label">{label}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def _render_line_chart(
    title: str,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    series_key: str,
) -> str:
    if not rows:
        return ""

    width = 900
    height = 320
    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    x_values = sorted({row[x_key] for row in rows})
    series_values = sorted({row[series_key] for row in rows})
    if not x_values:
        return ""

    min_x = min(x_values)
    max_x = max(x_values)
    x_span = max(max_x - min_x, 1)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{margin_left}" y="24" class="chart-title">{escape(title)}</text>')
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" '
        f'x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" class="axis" />'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + chart_height}" class="axis" />'
    )

    for tick in range(0, 6):
        value = tick / 5
        y = margin_top + chart_height - value * chart_height
        parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" class="grid" />'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" class="axis-label">{value:.1f}</text>'
        )

    for x_value in x_values:
        x = margin_left + ((x_value - min_x) / x_span) * chart_width
        parts.append(
            f'<text x="{x}" y="{margin_top + chart_height + 18}" text-anchor="middle" class="axis-label">{x_value}</text>'
        )

    for series_index, series_value in enumerate(series_values):
        series_rows = [row for row in rows if row[series_key] == series_value and row.get(y_key) is not None]
        if not series_rows:
            continue
        color = palette[series_index % len(palette)]
        series_rows = sorted(series_rows, key=lambda row: row[x_key])
        path_parts: list[str] = []
        for point_index, row in enumerate(series_rows):
            x = margin_left + ((row[x_key] - min_x) / x_span) * chart_width
            y = margin_top + chart_height - max(0.0, min(1.0, float(row[y_key]))) * chart_height
            command = "M" if point_index == 0 else "L"
            path_parts.append(f"{command} {x:.1f} {y:.1f}")
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
                f'<title>{escape(str(series_value))}: {x_key}={row[x_key]}, {y_key}={row[y_key]:.3f}</title>'
                "</circle>"
            )
        parts.append(f'<path d="{" ".join(path_parts)}" fill="none" stroke="{color}" stroke-width="2.5" />')

    legend_y = height - 12
    for series_index, series_value in enumerate(series_values):
        legend_x = margin_left + series_index * 150
        color = palette[series_index % len(palette)]
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />')
        parts.append(
            f'<text x="{legend_x + 18}" y="{legend_y}" class="legend-label">{escape(str(series_value))}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def render_html_report(report: dict[str, Any], *, input_paths: list[str]) -> str:
    overall = report["overall"]
    overview_rows = [
        [
            escape(row["model"]),
            _format_metric(row["detection_accuracy"], percent=True),
            _format_metric(row["joint_accuracy"], percent=True),
            _format_metric(row["localization_accuracy_on_error_rows"], percent=True),
            _format_metric(row["false_positive_rate_clean"], percent=True),
            _format_metric(row["latency_mean_seconds"]),
            _format_metric(row["total_tokens_mean"]),
        ]
        for row in report["by_model"]
    ]

    by_model_difficulty = report["by_model_difficulty"]
    difficulty_rows = [row for row in by_model_difficulty]
    error_type_rows = [row for row in report["by_model_error_type"]]
    step_rows = [row for row in report["by_model_expected_step"]]

    overview_table = _render_table(
        [
            "Model",
            "Detection",
            "Joint",
            "Localization",
            "Clean FPR",
            "Mean Latency (s)",
            "Mean Tokens",
        ],
        overview_rows,
    )

    summary_cards = _render_table(
        ["Metric", "Value"],
        [
            ["Total rows", str(report["total_rows"])],
            ["Detection accuracy", _format_metric(overall["detection_accuracy"], percent=True)],
            ["Joint accuracy", _format_metric(overall["joint_accuracy"], percent=True)],
            [
                "Localization on error rows",
                _format_metric(overall["localization_accuracy_on_error_rows"], percent=True),
            ],
            ["False positive rate on clean rows", _format_metric(overall["false_positive_rate_clean"], percent=True)],
            ["False negative rate on error rows", _format_metric(overall["false_negative_rate_error"], percent=True)],
        ],
    )

    charts = [
        _render_bar_chart(
            "Accuracy By Model",
            report["by_model"],
            metrics=[
                ("detection_accuracy", "Detection"),
                ("joint_accuracy", "Joint"),
                ("localization_accuracy_on_error_rows", "Localization"),
            ],
        ),
        _render_heatmap(
            "Joint Accuracy By Difficulty",
            difficulty_rows,
            row_key="difficulty",
            column_key="model",
            metric_key="joint_accuracy",
            row_order=["easy", "medium", "hard", "expert"],
            column_order=report["models"],
        ),
        _render_heatmap(
            "Joint Accuracy By Error Type",
            error_type_rows,
            row_key="error_type",
            column_key="model",
            metric_key="joint_accuracy",
            row_order=["clean", "E-RES", "E-INV", "E-CASC"],
            column_order=report["models"],
        ),
        _render_line_chart(
            "Localization By True Error Step",
            step_rows,
            x_key="expected_error_step",
            y_key="localization_accuracy_on_error_rows",
            series_key="model",
        ),
    ]

    input_items = "".join(f"<li>{escape(path)}</li>" for path in input_paths)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Emoji-Bench Evaluation Report</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --panel: #fffdf8;
      --ink: #221f1a;
      --muted: #6d655d;
      --line: #ddd0bf;
      --accent: #ae5f2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0, transparent 28%),
        linear-gradient(180deg, #fbf7ee 0, var(--bg) 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.5;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      margin-top: 18px;
      box-shadow: 0 10px 30px rgba(84, 57, 36, 0.06);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ece3d8;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      color: var(--ink);
      font-weight: 700;
    }}
    td {{
      color: var(--muted);
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: #fff;
      border: 1px solid #ece3d8;
      border-radius: 14px;
      padding: 6px;
    }}
    .chart-title {{
      font-size: 16px;
      font-weight: 700;
      fill: var(--ink);
    }}
    .axis {{
      stroke: #8e8377;
      stroke-width: 1.3;
    }}
    .grid line, line.grid {{
      stroke: #ece3d8;
      stroke-width: 1;
    }}
    .axis-label, .legend-label, .cell-label {{
      font-size: 11px;
      fill: #5e574f;
    }}
    .cell-label {{
      font-size: 12px;
      font-weight: 700;
    }}
    .lede {{
      max-width: 72ch;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #f2e1d0;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    ul {{
      margin: 8px 0 0 18px;
    }}
  </style>
</head>
<body>
  <main>
    <div class="badge">Emoji-Bench Eval Report</div>
    <h1>Model Comparison Dashboard</h1>
    <p class="lede">
      This report summarizes detection, localization, and cost/latency tradeoffs across one or more evaluation runs.
      It focuses on metrics that matter for Emoji-Bench specifically: finding whether an error exists, then finding the first wrong step.
    </p>

    <div class="grid">
      <section class="panel">
        <h2>Overview</h2>
        {summary_cards}
      </section>
      <section class="panel">
        <h2>Inputs</h2>
        <ul>{input_items}</ul>
      </section>
    </div>

    <section class="panel">
      <h2>By Model</h2>
      {overview_table}
    </section>

    <section class="panel">
      <h2>Charts</h2>
      {''.join(chart for chart in charts if chart)}
    </section>
  </main>
</body>
</html>
"""
