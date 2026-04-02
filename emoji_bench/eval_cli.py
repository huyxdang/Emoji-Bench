from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from emoji_bench.evaluation import (
    append_jsonl,
    load_jsonl_records,
    normalize_prediction,
    score_prediction,
    scored_prediction_to_dict,
    summarize_scores,
)
from emoji_bench.model_registry import (
    ModelConfig,
    ProviderName,
    get_model_config,
    list_model_configs,
    model_choices,
)
from emoji_bench.provider_eval import request_prediction, resolve_api_key, make_client


def _single_allowed_provider(
    allowed_providers: tuple[ProviderName, ...] | None,
) -> ProviderName | None:
    if allowed_providers is None or len(allowed_providers) != 1:
        return None
    return allowed_providers[0]


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


def _default_output_dir(input_path: Path, model_key: str) -> Path:
    dataset_name = input_path.parent.name if input_path.name == "test.jsonl" else input_path.stem
    return Path("artifacts") / "evals" / f"{dataset_name}-{_slugify(model_key)}"


def _shard_label(shard_index: int, num_shards: int) -> str:
    width = max(2, len(str(num_shards - 1)))
    return f"shard-{shard_index:0{width}d}-of-{num_shards:0{width}d}"


def _resolve_output_dir(
    *,
    raw_output_dir: str | None,
    input_path: Path,
    model_key: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    base_output_dir = (
        Path(raw_output_dir)
        if raw_output_dir is not None
        else _default_output_dir(input_path, model_key)
    )
    if num_shards == 1:
        return base_output_dir

    shard_dir_name = _shard_label(shard_index, num_shards)
    if base_output_dir.name == shard_dir_name:
        return base_output_dir
    return base_output_dir / shard_dir_name


def _validate_sharding_args(
    parser: argparse.ArgumentParser,
    *,
    num_shards: int,
    shard_index: int,
) -> None:
    if num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if shard_index < 0:
        parser.error("--shard-index must be >= 0")
    if shard_index >= num_shards:
        parser.error("--shard-index must be less than --num-shards")


def _stable_shard_index(example_id: str, *, num_shards: int) -> int:
    digest = hashlib.blake2b(example_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def _select_shard_records(
    records: list[dict[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if num_shards == 1:
        return records
    return [
        record
        for record in records
        if _stable_shard_index(record["example_id"], num_shards=num_shards) == shard_index
    ]


def _load_existing_scores(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    records = load_jsonl_records(path)
    seen = {record["example_id"] for record in records}
    return seen, records


def build_parser(
    *,
    description: str,
    allowed_providers: tuple[ProviderName, ...] | None,
    default_model: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    single_provider = _single_allowed_provider(allowed_providers)
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a dataset JSONL file, or a dataset directory containing test.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        choices=model_choices(providers=allowed_providers),
        help="Configured model alias to evaluate.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available model configs as JSON and exit.",
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
        "--num-shards",
        type=int,
        default=1,
        help="Split the dataset into this many stable shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to evaluate.",
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
        default=None,
        help="Optional override for the configured default max output tokens.",
    )
    if single_provider in {None, "openai"}:
        parser.add_argument(
            "--reasoning-effort",
            choices=("none", "minimal", "low", "medium", "high", "xhigh"),
            default=None,
            help="Optional override for OpenAI reasoning models.",
        )
    else:
        parser.set_defaults(reasoning_effort=None)

    if single_provider in {None, "anthropic"}:
        parser.add_argument(
            "--thinking-budget-tokens",
            type=int,
            default=None,
            help="Optional Anthropic extended thinking budget. Must be less than max output tokens.",
        )
    else:
        parser.set_defaults(thinking_budget_tokens=None)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional provider API key. Defaults to the environment variable for the selected model.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing predictions.jsonl file.",
    )
    return parser


def main(
    *,
    description: str,
    allowed_providers: tuple[ProviderName, ...] | None = None,
    default_model: str,
) -> None:
    parser = build_parser(
        description=description,
        allowed_providers=allowed_providers,
        default_model=default_model,
    )
    args = parser.parse_args()

    if args.list_models:
        configs = [
            config.to_dict()
            for config in list_model_configs()
            if allowed_providers is None or config.provider in set(allowed_providers)
        ]
        print(json.dumps(configs, ensure_ascii=False, indent=2))
        return

    if args.input_path is None:
        parser.error("input_path is required unless --list-models is used")

    _validate_sharding_args(
        parser,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")

    model_config = get_model_config(args.model)
    if allowed_providers is not None and model_config.provider not in set(allowed_providers):
        parser.error(f"{model_config.key} is not available in this evaluator wrapper")

    if args.reasoning_effort is not None and model_config.provider != "openai":
        parser.error("--reasoning-effort only applies to OpenAI models")
    if args.thinking_budget_tokens is not None and model_config.provider != "anthropic":
        parser.error("--thinking-budget-tokens only applies to Anthropic models")

    api_key = resolve_api_key(
        model_config=model_config,
        explicit_api_key=args.api_key,
        env=os.environ,
    )

    input_path = _resolve_input_path(args.input_path)
    records = load_jsonl_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]
    records = _select_shard_records(
        records,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    max_output_tokens = args.max_output_tokens or model_config.default_max_output_tokens
    output_dir = _resolve_output_dir(
        raw_output_dir=args.output_dir,
        input_path=input_path,
        model_key=model_config.key,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
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

    client = make_client(model_config.provider, api_key=api_key)
    scored_records = list(existing_score_records)

    for record in records:
        if record["example_id"] in seen_example_ids:
            continue

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                request_started = time.perf_counter()
                provider_response = request_prediction(
                    client=client,
                    model_config=model_config,
                    prompt=record["prompt"],
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                )
                request_latency_seconds = time.perf_counter() - request_started
                prediction = normalize_prediction(provider_response.prediction_payload)
                scored = score_prediction(record, prediction)
                row = scored_prediction_to_dict(scored)
                row["model"] = model_config.key
                row["provider"] = model_config.provider
                row["api_model"] = model_config.api_model
                row["response_id"] = provider_response.response_id
                row["raw_prediction"] = provider_response.prediction_payload
                row["raw_output_text"] = provider_response.raw_output_text
                row["request_latency_seconds"] = request_latency_seconds
                usage = provider_response.usage
                row["input_tokens"] = None if usage is None else usage.input_tokens
                row["output_tokens"] = None if usage is None else usage.output_tokens
                row["reasoning_tokens"] = None if usage is None else usage.reasoning_tokens
                row["total_tokens"] = None if usage is None else usage.total_tokens
                row["num_shards"] = args.num_shards
                row["shard_index"] = args.shard_index
                row["shard_label"] = _shard_label(args.shard_index, args.num_shards)
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
            "model": model_config.key,
            "provider": model_config.provider,
            "api_model": model_config.api_model,
            "input_path": str(input_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "predictions_path": str(predictions_path.resolve()),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "shard_label": _shard_label(args.shard_index, args.num_shards),
            "shard_total_examples": len(records),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
