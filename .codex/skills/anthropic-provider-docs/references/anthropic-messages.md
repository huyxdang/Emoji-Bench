# Anthropic Messages API

Official sources checked on April 2, 2026:

- Messages API: `https://docs.anthropic.com/en/api/messages`
- Structured outputs: `https://platform.claude.com/docs/en/build-with-claude/structured-outputs`
- Extended thinking tips: `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/extended-thinking-tips`

Key details used by Emoji-Bench:

- The Messages API uses top-level `model`, `system`, `messages`, and `max_tokens`.
- Anthropic structured outputs support `output_config.format` with a JSON schema.
- Anthropic docs state that valid JSON matching the schema is returned in `response.content[0].text`.
- Extended thinking uses a `thinking` object with `type: "enabled"` and `budget_tokens`.
- Anthropic docs recommend starting with the minimum thinking budget and increasing only if needed.
- Anthropic docs state the minimum thinking budget is `1024` tokens.

Repo mapping:

- `emoji_bench/provider_eval.py` uses `client.messages.create(...)`.
- Structured output is parsed from the returned text block as JSON.
- `emoji_bench/eval_cli.py` exposes `--thinking-budget-tokens` only on Anthropic-capable entrypoints.
- Emoji-Bench's reasoning-labeled Sonnet 4.6 variant relies on the same `thinking` object and defaults to the minimum 1024-token budget when no CLI override is provided.
