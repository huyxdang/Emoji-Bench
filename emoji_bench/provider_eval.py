from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from emoji_bench.model_registry import ModelConfig, ProviderName, ReasoningEffort


SYSTEM_PROMPT = """\
You are reviewing your own prior working out in a formal system.

Return only the structured fields requested by the schema:
- has_error: true if there is any incorrect step in the working out, otherwise false
- error_step: the first incorrect step number, or null if there is no error

Do not explain your answer."""


PREDICTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "has_error": {"type": "boolean"},
        "error_step": {"type": ["integer", "null"]},
    },
    "required": ["has_error", "error_step"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ProviderResponse:
    prediction_payload: dict[str, Any]
    response_id: str | None
    raw_output_text: str
    usage: ProviderUsage | None = None


def build_openai_request_options(
    *,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: ReasoningEffort | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": model_config.api_model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": max_output_tokens,
    }

    resolved_effort = reasoning_effort
    if resolved_effort is None and model_config.openai_reasoning is not None:
        resolved_effort = model_config.openai_reasoning.effort

    if resolved_effort is not None:
        reasoning: dict[str, str] = {"effort": resolved_effort}
        if model_config.openai_reasoning is not None and model_config.openai_reasoning.summary:
            reasoning["summary"] = model_config.openai_reasoning.summary
        options["reasoning"] = reasoning

    return options


def build_anthropic_request_options(
    *,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": model_config.api_model,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": PREDICTION_JSON_SCHEMA,
            }
        },
    }

    resolved_budget = thinking_budget_tokens
    if resolved_budget is None and model_config.anthropic_thinking is not None:
        if model_config.anthropic_thinking.enabled:
            resolved_budget = model_config.anthropic_thinking.budget_tokens

    if resolved_budget is not None:
        if resolved_budget >= max_output_tokens:
            raise ValueError("Anthropic thinking budget must be less than max_output_tokens")
        options["thinking"] = {
            "type": "enabled",
            "budget_tokens": resolved_budget,
        }

    return options


def resolve_api_key(
    *,
    model_config: ModelConfig,
    explicit_api_key: str | None,
    env: dict[str, str],
) -> str:
    api_key = explicit_api_key or env.get(model_config.api_key_env_var)
    if api_key:
        return api_key
    raise RuntimeError(
        f"{model_config.api_key_env_var} is required for {model_config.key}. "
        "Set it in the environment or pass --api-key."
    )


def make_client(provider: ProviderName, *, api_key: str) -> Any:
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenAI evaluation. "
                'Install with `pip install -e ".[openai]"`.'
            ) from exc
        return OpenAI(api_key=api_key)

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is required for Anthropic evaluation. "
                'Install with `pip install -e ".[anthropic]"`.'
            ) from exc
        return Anthropic(api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")


def request_prediction(
    *,
    client: Any,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: ReasoningEffort | None = None,
    thinking_budget_tokens: int | None = None,
) -> ProviderResponse:
    if model_config.provider == "openai":
        return _request_openai_prediction(
            client=client,
            model_config=model_config,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
    if model_config.provider == "anthropic":
        return _request_anthropic_prediction(
            client=client,
            model_config=model_config,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    raise ValueError(f"Unsupported provider: {model_config.provider}")


def _request_openai_prediction(
    *,
    client: Any,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: ReasoningEffort | None,
) -> ProviderResponse:
    PredictionModel = _make_prediction_model()
    options = build_openai_request_options(
        model_config=model_config,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    response = client.responses.parse(
        text_format=PredictionModel,
        **options,
    )

    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            payload = parsed.model_dump()
        elif hasattr(parsed, "dict"):
            payload = parsed.dict()
        else:
            payload = dict(parsed)
        return ProviderResponse(
            prediction_payload=payload,
            response_id=getattr(response, "id", None),
            raw_output_text=getattr(response, "output_text", ""),
            usage=_extract_openai_usage(response),
        )

    output_text = getattr(response, "output_text", "")
    if output_text:
        return ProviderResponse(
            prediction_payload=json.loads(output_text),
            response_id=getattr(response, "id", None),
            raw_output_text=output_text,
            usage=_extract_openai_usage(response),
        )

    raise ValueError("No structured output returned by the OpenAI model")


def _request_anthropic_prediction(
    *,
    client: Any,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
    thinking_budget_tokens: int | None,
) -> ProviderResponse:
    options = build_anthropic_request_options(
        model_config=model_config,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    response = client.messages.create(**options)

    raw_output_text = _anthropic_text_output(response)
    if raw_output_text:
        return ProviderResponse(
            prediction_payload=json.loads(raw_output_text),
            response_id=getattr(response, "id", None),
            raw_output_text=raw_output_text,
            usage=_extract_anthropic_usage(response),
        )

    raise ValueError("No structured output returned by the Anthropic model")


def _anthropic_text_output(response: Any) -> str:
    blocks = getattr(response, "content", None) or ()
    texts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            texts.append(block.text)
    return "\n".join(texts).strip()


def _make_prediction_model():
    from pydantic import BaseModel

    class ErrorCheckPrediction(BaseModel):
        has_error: bool
        error_step: int | None

    return ErrorCheckPrediction


def _extract_openai_usage(response: Any) -> ProviderUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    reasoning_tokens = None

    output_details = getattr(usage, "output_tokens_details", None)
    if output_details is not None:
        reasoning_tokens = getattr(output_details, "reasoning_tokens", None)

    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def _extract_anthropic_usage(response: Any) -> ProviderUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=None,
        total_tokens=total_tokens,
    )
