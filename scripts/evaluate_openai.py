#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.evaluation import (
    append_jsonl,
    load_jsonl_records,
    normalize_prediction,
    score_prediction,
    scored_prediction_to_dict,
    summarize_scores,
)


SYSTEM_PROMPT = """\
You are reviewing your own prior working out in a formal system.

Return only the structured fields requested by the schema:
- has_error: true if there is any incorrect step in the working out, otherwise false
- error_step: the first incorrect step number, or null if there is no error

Do not explain your answer."""


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "test.jsonl"
    return path


def _default_output_dir(input_path: Path, model: str) -> Path:
    dataset_name = input_path.parent.name if input_path.name == "test.jsonl" else input_path.stem
    return Path("artifacts") / "evals" / f"{dataset_name}-{_slugify(model)}"


def _load_existing_scores(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    records = load_jsonl_records(path)
    seen = {record["example_id"] for record in records}
    return seen, records


def _make_prediction_model():
    from pydantic import BaseModel

    class ErrorCheckPrediction(BaseModel):
        has_error: bool
        error_step: int | None

    return ErrorCheckPrediction


def _request_prediction(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], Any]:
    PredictionModel = _make_prediction_model()
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text_format=PredictionModel,
        max_output_tokens=max_output_tokens,
    )

    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump(), response
        if hasattr(parsed, "dict"):
            return parsed.dict(), response
        return dict(parsed), response

    output_text = getattr(response, "output_text", "")
    if output_text:
        return json.loads(output_text), response

    raise ValueError("No structured output returned by the model")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an OpenAI model on Emoji-Bench prompts and score has_error / error_step.",
    )
    parser.add_argument(
        "input_path",
        help="Path to a dataset JSONL file, or a dataset directory containing test.jsonl.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to evaluate. Defaults to gpt-4.1-mini.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for predictions and summary outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to evaluate.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per example on API or parsing failures.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Delay between retries after a failed request.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=50,
        help="Max output tokens per model response.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional OpenAI API key. Defaults to OPENAI_API_KEY from the environment or .env.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing predictions.jsonl file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Set it in the environment or .env.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for this evaluator. "
            'Install with `pip install -e ".[openai]"`.'
        ) from exc

    input_path = _resolve_input_path(args.input_path)
    records = load_jsonl_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(input_path, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if args.no_resume:
        seen_example_ids: set[str] = set()
        existing_score_records: list[dict[str, Any]] = []
        if predictions_path.exists():
            predictions_path.unlink()
    else:
        seen_example_ids, existing_score_records = _load_existing_scores(predictions_path)

    client = OpenAI(api_key=api_key)
    scored_records = list(existing_score_records)

    for index, record in enumerate(records, start=1):
        if record["example_id"] in seen_example_ids:
            continue

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                prediction_payload, response = _request_prediction(
                    client=client,
                    model=args.model,
                    prompt=record["prompt"],
                    max_output_tokens=args.max_output_tokens,
                )
                prediction = normalize_prediction(prediction_payload)
                scored = score_prediction(record, prediction)
                row = scored_prediction_to_dict(scored)
                row["model"] = args.model
                row["response_id"] = getattr(response, "id", None)
                row["raw_prediction"] = prediction_payload
                row["raw_output_text"] = getattr(response, "output_text", "")
                append_jsonl(predictions_path, row)
                scored_records.append(row)
                seen_example_ids.add(record["example_id"])
                print(
                    f"[{len(seen_example_ids)}/{len(records)}] {record['example_id']} "
                    f"joint={row['joint_correct']}"
                )
                if args.request_delay_seconds > 0:
                    time.sleep(args.request_delay_seconds)
                break
            except Exception as exc:
                last_error = exc
                if attempt == args.max_retries:
                    raise
                time.sleep(args.retry_delay_seconds)

        if last_error is not None and record["example_id"] not in seen_example_ids:
            raise last_error

    summary = summarize_scores(
        [
            score_prediction(
                {
                    "example_id": row["example_id"],
                    "difficulty": row["difficulty"],
                    "error_type": row["error_type"],
                    "has_error": row["expected_has_error"],
                    "expected_error_step": row["expected_error_step"],
                },
                normalize_prediction(
                    {
                        "has_error": row["predicted_has_error"],
                        "error_step": row["predicted_error_step"],
                    }
                ),
            )
            for row in scored_records
        ]
    )
    summary.update(
        {
            "model": args.model,
            "input_path": str(input_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "predictions_path": str(predictions_path.resolve()),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
