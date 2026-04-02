---
name: openai-provider-docs
description: Use when changing Emoji-Bench's OpenAI evaluator integration, GPT-5.4 model registry entries, or Responses API request shape. Covers official OpenAI docs for Responses API, structured outputs, and reasoning.effort.
---

# OpenAI Provider Docs

Use this skill when editing the OpenAI side of Emoji-Bench's evaluator stack:

- `emoji_bench/model_registry.py`
- `emoji_bench/provider_eval.py`
- `emoji_bench/eval_cli.py`
- `scripts/evaluate_model.py`
- `scripts/evaluate_openai.py`

## Workflow

1. Read `references/openai-responses.md` before changing request payload shape.
2. Read `references/openai-models.md` before changing GPT-5.4 model aliases, limits, or defaults.
3. Prefer the Responses API over Chat Completions for this repo.
4. For GPT-5.x models, keep `reasoning.effort` separate from `max_output_tokens`.
5. If OpenAI docs change, update the references with the access date and the affected code paths.

## Repo Conventions

- GPT-5.4, GPT-5.4 mini, and GPT-5.4 nano are configured with `reasoning.effort="medium"` for evaluator runs.
- The evaluator still keeps `gpt-4.1-mini` as a legacy baseline for backward compatibility.
- The OpenAI wrapper script should stay usable with `python scripts/evaluate_openai.py ...`.
