---
name: anthropic-provider-docs
description: Use when changing Emoji-Bench's Anthropic evaluator integration, Claude Sonnet 4.6 or Claude Haiku 4.5 model registry entries, or Messages API request shape. Covers official Anthropic docs for Messages API, structured outputs, and extended thinking.
---

# Anthropic Provider Docs

Use this skill when editing the Anthropic side of Emoji-Bench's evaluator stack:

- `emoji_bench/model_registry.py`
- `emoji_bench/provider_eval.py`
- `emoji_bench/eval_cli.py`
- `scripts/evaluate_model.py`
- `scripts/evaluate_anthropic.py`

## Workflow

1. Read `references/anthropic-messages.md` before changing request payload shape.
2. Read `references/anthropic-models.md` before changing Claude aliases, limits, or defaults.
3. Prefer the Messages API for this repo's Anthropic integration.
4. Keep `max_tokens` separate from any extended-thinking budget.
5. If Anthropic docs change, update the references with the access date and the affected code paths.

## Repo Conventions

- `claude-sonnet-4-6` and `claude-haiku-4-5` are the configured Anthropic aliases.
- Anthropic structured output is implemented with `output_config.format` and a JSON schema.
- Extended thinking is supported by the configured Claude 4.x models but is disabled by default in the evaluator.
