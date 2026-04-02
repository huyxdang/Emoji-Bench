from emoji_bench.model_registry import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONFIGS,
    get_model_config,
    model_choices,
)
from emoji_bench.provider_eval import (
    PREDICTION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_anthropic_request_options,
    build_openai_request_options,
    resolve_api_key,
)


def test_requested_model_configs_are_present():
    assert {
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    }.issubset(set(model_choices()))


def test_gpt54_models_default_to_medium_reasoning():
    for key in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"):
        config = get_model_config(key)
        assert config.provider == "openai"
        assert config.openai_reasoning is not None
        assert config.openai_reasoning.effort == "medium"
        assert config.default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_all_configured_models_default_to_512_max_output_tokens():
    assert DEFAULT_MAX_OUTPUT_TOKENS == 512
    for config in MODEL_CONFIGS.values():
        assert config.default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_build_openai_request_options_uses_reasoning_config():
    config = get_model_config("gpt-5.4-mini")
    options = build_openai_request_options(
        model_config=config,
        prompt="example prompt",
        max_output_tokens=77,
    )

    assert options["model"] == "gpt-5.4-mini"
    assert options["max_output_tokens"] == 77
    assert options["reasoning"] == {"effort": "medium"}
    assert options["input"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert options["input"][1] == {"role": "user", "content": "example prompt"}


def test_build_anthropic_request_options_includes_json_schema_and_optional_thinking():
    config = get_model_config("claude-sonnet-4-6")
    options = build_anthropic_request_options(
        model_config=config,
        prompt="example prompt",
        max_output_tokens=77,
        thinking_budget_tokens=32,
    )

    assert options["model"] == "claude-sonnet-4-6"
    assert options["system"] == SYSTEM_PROMPT
    assert options["messages"] == [{"role": "user", "content": "example prompt"}]
    assert options["max_tokens"] == 77
    assert options["output_config"]["format"]["schema"] == PREDICTION_JSON_SCHEMA
    assert options["thinking"] == {"type": "enabled", "budget_tokens": 32}


def test_build_anthropic_request_options_rejects_invalid_thinking_budget():
    config = get_model_config("claude-haiku-4-5")

    try:
        build_anthropic_request_options(
            model_config=config,
            prompt="example prompt",
            max_output_tokens=64,
            thinking_budget_tokens=64,
        )
    except ValueError as exc:
        assert "less than max_output_tokens" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected invalid Anthropic thinking budget to fail")


def test_resolve_api_key_uses_provider_specific_env_var():
    config = get_model_config("claude-sonnet-4-6")
    api_key = resolve_api_key(
        model_config=config,
        explicit_api_key=None,
        env={"ANTHROPIC_API_KEY": "test-key"},
    )
    assert api_key == "test-key"
