from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ProviderName = Literal["openai", "anthropic"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class OpenAIReasoningConfig:
    effort: ReasoningEffort
    summary: str | None = None


@dataclass(frozen=True)
class AnthropicThinkingConfig:
    enabled: bool
    budget_tokens: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    provider: ProviderName
    api_model: str
    docs_url: str
    api_key_env_var: str
    default_max_output_tokens: int
    provider_max_output_tokens: int | None = None
    openai_reasoning: OpenAIReasoningConfig | None = None
    anthropic_thinking: AnthropicThinkingConfig | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Backward-compatible OpenAI default used by the legacy evaluator wrapper.
    "gpt-4.1-mini": ModelConfig(
        key="gpt-4.1-mini",
        label="GPT-4.1 mini",
        provider="openai",
        api_model="gpt-4.1-mini",
        docs_url="https://developers.openai.com/api/docs/models",
        api_key_env_var="OPENAI_API_KEY",
        default_max_output_tokens=50,
        provider_max_output_tokens=None,
        notes="Legacy non-reasoning baseline kept for backward compatibility.",
    ),
    "gpt-5.4": ModelConfig(
        key="gpt-5.4",
        label="GPT-5.4",
        provider="openai",
        api_model="gpt-5.4",
        docs_url="https://developers.openai.com/api/docs/models/gpt-5.4",
        api_key_env_var="OPENAI_API_KEY",
        default_max_output_tokens=50,
        provider_max_output_tokens=128_000,
        openai_reasoning=OpenAIReasoningConfig(effort="medium"),
        notes="Configured to use medium reasoning effort for evaluation runs.",
    ),
    "gpt-5.4-mini": ModelConfig(
        key="gpt-5.4-mini",
        label="GPT-5.4 mini",
        provider="openai",
        api_model="gpt-5.4-mini",
        docs_url="https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        api_key_env_var="OPENAI_API_KEY",
        default_max_output_tokens=50,
        provider_max_output_tokens=128_000,
        openai_reasoning=OpenAIReasoningConfig(effort="medium"),
        notes="Configured to use medium reasoning effort for evaluation runs.",
    ),
    "gpt-5.4-nano": ModelConfig(
        key="gpt-5.4-nano",
        label="GPT-5.4 nano",
        provider="openai",
        api_model="gpt-5.4-nano",
        docs_url="https://developers.openai.com/api/docs/models/gpt-5.4-nano",
        api_key_env_var="OPENAI_API_KEY",
        default_max_output_tokens=50,
        provider_max_output_tokens=128_000,
        openai_reasoning=OpenAIReasoningConfig(effort="medium"),
        notes="Configured to use medium reasoning effort for evaluation runs.",
    ),
    "claude-sonnet-4-6": ModelConfig(
        key="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        provider="anthropic",
        api_model="claude-sonnet-4-6",
        docs_url="https://platform.claude.com/docs/en/about-claude/models/overview",
        api_key_env_var="ANTHROPIC_API_KEY",
        default_max_output_tokens=50,
        provider_max_output_tokens=64_000,
        anthropic_thinking=AnthropicThinkingConfig(enabled=False),
        notes="Extended thinking is supported by the model, but disabled by default in this evaluator.",
    ),
    "claude-haiku-4-5": ModelConfig(
        key="claude-haiku-4-5",
        label="Claude Haiku 4.5",
        provider="anthropic",
        api_model="claude-haiku-4-5",
        docs_url="https://platform.claude.com/docs/en/about-claude/models/overview",
        api_key_env_var="ANTHROPIC_API_KEY",
        default_max_output_tokens=50,
        provider_max_output_tokens=64_000,
        anthropic_thinking=AnthropicThinkingConfig(enabled=False),
        notes=(
            "Anthropic's official docs list claude-haiku-4-5 as the alias and "
            "claude-haiku-4-5-20251001 as the snapshot ID."
        ),
    ),
}


def get_model_config(key: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model config '{key}'. Expected one of: {known}") from exc


def list_model_configs(*, provider: ProviderName | None = None) -> list[ModelConfig]:
    configs = sorted(MODEL_CONFIGS.values(), key=lambda config: config.key)
    if provider is None:
        return configs
    return [config for config in configs if config.provider == provider]


def model_choices(*, providers: tuple[ProviderName, ...] | None = None) -> tuple[str, ...]:
    if providers is None:
        return tuple(config.key for config in list_model_configs())
    allowed = set(providers)
    return tuple(
        config.key
        for config in list_model_configs()
        if config.provider in allowed
    )
