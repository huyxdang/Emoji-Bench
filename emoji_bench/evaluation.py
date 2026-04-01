from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalPrediction:
    has_error: bool
    error_step: int | None


@dataclass(frozen=True)
class ScoredPrediction:
    example_id: str
    difficulty: str
    error_type: str | None
    expected_has_error: bool
    expected_error_step: int | None
    predicted_has_error: bool
    predicted_error_step: int | None
    has_error_correct: bool
    error_step_correct: bool
    joint_correct: bool


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_prediction(data: dict[str, Any]) -> EvalPrediction:
    if "has_error" not in data:
        raise ValueError("prediction must include 'has_error'")

    raw_has_error = data["has_error"]
    if isinstance(raw_has_error, bool):
        has_error = raw_has_error
    elif isinstance(raw_has_error, str):
        normalized = raw_has_error.strip().lower()
        if normalized in {"true", "yes"}:
            has_error = True
        elif normalized in {"false", "no"}:
            has_error = False
        else:
            raise ValueError(f"invalid has_error value: {raw_has_error!r}")
    else:
        raise ValueError(f"invalid has_error type: {type(raw_has_error).__name__}")

    raw_error_step = data.get("error_step")
    if raw_error_step is None or raw_error_step == "":
        error_step = None
    elif isinstance(raw_error_step, int):
        error_step = raw_error_step
    elif isinstance(raw_error_step, str) and raw_error_step.strip().isdigit():
        error_step = int(raw_error_step.strip())
    else:
        raise ValueError(f"invalid error_step value: {raw_error_step!r}")

    if error_step is not None and error_step < 1:
        raise ValueError("error_step must be >= 1 when provided")

    if not has_error:
        error_step = None

    return EvalPrediction(has_error=has_error, error_step=error_step)


def score_prediction(
    record: dict[str, Any],
    prediction: EvalPrediction,
) -> ScoredPrediction:
    expected_has_error = bool(record["has_error"])
    expected_error_step = record["expected_error_step"]

    has_error_correct = prediction.has_error == expected_has_error
    error_step_correct = prediction.error_step == expected_error_step
    joint_correct = has_error_correct and error_step_correct

    return ScoredPrediction(
        example_id=record["example_id"],
        difficulty=record["difficulty"],
        error_type=record["error_type"],
        expected_has_error=expected_has_error,
        expected_error_step=expected_error_step,
        predicted_has_error=prediction.has_error,
        predicted_error_step=prediction.error_step,
        has_error_correct=has_error_correct,
        error_step_correct=error_step_correct,
        joint_correct=joint_correct,
    )


def scored_prediction_to_dict(scored: ScoredPrediction) -> dict[str, Any]:
    return asdict(scored)


def summarize_scores(scored_predictions: list[ScoredPrediction]) -> dict[str, Any]:
    total = len(scored_predictions)
    if total == 0:
        return {
            "total_examples": 0,
            "has_error_accuracy": 0.0,
            "error_step_accuracy": 0.0,
            "joint_accuracy": 0.0,
            "by_difficulty": {},
        }

    def _accuracy(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_difficulty: dict[str, dict[str, Any]] = {}
    difficulties = sorted({scored.difficulty for scored in scored_predictions})
    for difficulty in difficulties:
        subset = [scored for scored in scored_predictions if scored.difficulty == difficulty]
        by_difficulty[difficulty] = {
            "total_examples": len(subset),
            "has_error_accuracy": _accuracy([s.has_error_correct for s in subset]),
            "error_step_accuracy": _accuracy([s.error_step_correct for s in subset]),
            "joint_accuracy": _accuracy([s.joint_correct for s in subset]),
        }

    return {
        "total_examples": total,
        "has_error_accuracy": _accuracy([s.has_error_correct for s in scored_predictions]),
        "error_step_accuracy": _accuracy([s.error_step_correct for s in scored_predictions]),
        "joint_accuracy": _accuracy([s.joint_correct for s in scored_predictions]),
        "by_difficulty": by_difficulty,
    }
