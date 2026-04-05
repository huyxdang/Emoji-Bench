from emoji_bench.model_registry import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONFIGS,
    get_model_config,
    model_choices,
)
from emoji_bench.provider_eval import (
    GEMINI_PREDICTION_JSON_SCHEMA,
    PREDICTION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_anthropic_request_options,
    build_gemini_request_options,
    build_mistral_request_options,
    build_openai_request_options,
    resolve_api_key,
)


def test_requested_model_configs_are_present():
    assert {
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-6-reasoning",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "mistral-large-2512",
        "mistral-medium-2508",
    }.issubset(set(model_choices()))


def test_gpt54_models_default_to_medium_reasoning():
    for key in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"):
        config = get_model_config(key)
        assert config.provider == "openai"
        assert config.openai_reasoning is not None
        assert config.openai_reasoning.effort == "medium"
    assert get_model_config("gpt-5.4").default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert get_model_config("gpt-5.4-mini").default_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert get_model_config("gpt-5.4-nano").default_max_output_tokens == 2048


def test_all_configured_models_use_expected_default_max_output_tokens():
    assert DEFAULT_MAX_OUTPUT_TOKENS == 2048
    for config in MODEL_CONFIGS.values():
        expected = DEFAULT_MAX_OUTPUT_TOKENS
        assert config.default_max_output_tokens == expected


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


def test_reasoning_sonnet_enables_anthropic_thinking_by_default():
    config = get_model_config("claude-sonnet-4-6-reasoning")
    options = build_anthropic_request_options(
        model_config=config,
        prompt="example prompt",
        max_output_tokens=2048,
    )

    assert config.label == "Claude Sonnet 4.6 (reasoning)"
    assert options["model"] == "claude-sonnet-4-6"
    assert options["thinking"] == {"type": "enabled", "budget_tokens": 1024}


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


def test_build_mistral_request_options_uses_json_mode():
    config = get_model_config("mistral-large-2512")
    options = build_mistral_request_options(
        model_config=config,
        prompt="example prompt",
        max_output_tokens=77,
    )

    assert options["model"] == "mistral-large-2512"
    assert options["max_tokens"] == 77
    assert options["temperature"] == 0
    assert options["response_format"] == {"type": "json_object"}
    assert options["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert options["messages"][1] == {"role": "user", "content": "example prompt"}


def test_build_gemini_request_options_uses_json_schema_mode():
    config = get_model_config("gemini-3.1-pro-preview")
    options = build_gemini_request_options(
        model_config=config,
        prompt="example prompt",
        max_output_tokens=77,
    )

    assert options["systemInstruction"] == {"parts": [{"text": SYSTEM_PROMPT}]}
    assert options["contents"] == [{"role": "user", "parts": [{"text": "example prompt"}]}]
    assert options["generationConfig"]["maxOutputTokens"] == 77
    assert options["generationConfig"]["responseMimeType"] == "application/json"
    assert options["generationConfig"]["responseJsonSchema"] == GEMINI_PREDICTION_JSON_SCHEMA


def test_resolve_api_key_uses_provider_specific_env_var():
    config = get_model_config("claude-sonnet-4-6")
    api_key = resolve_api_key(
        model_config=config,
        explicit_api_key=None,
        env={"ANTHROPIC_API_KEY": "test-key"},
    )
    assert api_key == "test-key"


def test_resolve_api_key_supports_gemini_env_var():
    config = get_model_config("gemini-3-flash-preview")
    api_key = resolve_api_key(
        model_config=config,
        explicit_api_key=None,
        env={"GEMINI_API_KEY": "test-gemini-key"},
    )
    assert api_key == "test-gemini-key"
