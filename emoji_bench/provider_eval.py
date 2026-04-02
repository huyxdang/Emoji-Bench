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

OPENAI_STRUCTURED_OUTPUT_RETRY_FLOOR = 200


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
    response = _openai_parse_prediction(
        client=client,
        model_config=model_config,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        prediction_model=PredictionModel,
    )

    payload = _extract_openai_prediction_payload(response)
    if payload is not None:
        return ProviderResponse(
            prediction_payload=payload,
            response_id=getattr(response, "id", None),
            raw_output_text=_openai_output_text(response),
            usage=_extract_openai_usage(response),
        )

    retry_max_output_tokens = _openai_retry_max_output_tokens(
        response=response,
        max_output_tokens=max_output_tokens,
        model_config=model_config,
    )
    if retry_max_output_tokens is not None:
        response = _openai_parse_prediction(
            client=client,
            model_config=model_config,
            prompt=prompt,
            max_output_tokens=retry_max_output_tokens,
            reasoning_effort=reasoning_effort,
            prediction_model=PredictionModel,
        )
        payload = _extract_openai_prediction_payload(response)
        if payload is not None:
            return ProviderResponse(
                prediction_payload=payload,
                response_id=getattr(response, "id", None),
                raw_output_text=_openai_output_text(response),
                usage=_extract_openai_usage(response),
            )

    raise ValueError(_openai_missing_output_error(response=response))


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


def _openai_parse_prediction(
    *,
    client: Any,
    model_config: ModelConfig,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: ReasoningEffort | None,
    prediction_model: type[Any],
) -> Any:
    options = build_openai_request_options(
        model_config=model_config,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    return client.responses.parse(
        text_format=prediction_model,
        **options,
    )


def _extract_openai_prediction_payload(response: Any) -> dict[str, Any] | None:
    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        return _coerce_parsed_payload(parsed)

    for output in getattr(response, "output", ()) or ():
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", ()) or ():
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return _coerce_parsed_payload(parsed)

    output_text = _openai_output_text(response)
    if output_text:
        return json.loads(output_text)

    return None


def _coerce_parsed_payload(parsed: Any) -> dict[str, Any]:
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if hasattr(parsed, "dict"):
        return parsed.dict()
    return dict(parsed)


def _openai_output_text(response: Any) -> str:
    direct_output_text = getattr(response, "output_text", "")
    if direct_output_text:
        return direct_output_text

    parts: list[str] = []
    for output in getattr(response, "output", ()) or ():
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", ()) or ():
            if getattr(content, "type", None) == "output_text" and hasattr(content, "text"):
                text = content.text.strip()
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _openai_retry_max_output_tokens(
    *,
    response: Any,
    max_output_tokens: int,
    model_config: ModelConfig,
) -> int | None:
    incomplete_details = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete_details, "reason", None)
    if reason != "max_output_tokens":
        return None

    provider_limit = model_config.provider_max_output_tokens
    retry_tokens = max(max_output_tokens * 2, OPENAI_STRUCTURED_OUTPUT_RETRY_FLOOR)
    if provider_limit is not None:
        retry_tokens = min(retry_tokens, provider_limit)

    if retry_tokens <= max_output_tokens:
        return None
    return retry_tokens


def _openai_missing_output_error(*, response: Any) -> str:
    status = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None)
    output_types = [
        getattr(output, "type", type(output).__name__)
        for output in getattr(response, "output", ()) or ()
    ]

    details: list[str] = []
    if status is not None:
        details.append(f"status={status}")
    if incomplete_reason is not None:
        details.append(f"incomplete_reason={incomplete_reason}")
    if output_types:
        details.append(f"output_types={output_types}")

    message = "No structured output returned by the OpenAI model"
    if details:
        message += " (" + ", ".join(details) + ")"
    if incomplete_reason == "max_output_tokens":
        message += ". Retry with a higher --max-output-tokens value."
    return message


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
