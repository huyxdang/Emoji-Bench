# OpenAI Responses API

Official sources checked on April 2, 2026:

- Responses API reference: `https://platform.openai.com/docs/api-reference/responses/compact?api-mode=responses`
- Structured outputs guide: `https://developers.openai.com/api/docs/guides/structured-outputs`

Key details used by Emoji-Bench:

- The Responses API accepts `model`, `input`, and `max_output_tokens`.
- Reasoning models accept a `reasoning` object.
- `reasoning.effort` currently supports `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
- OpenAI docs state that `reasoning.effort` defaults to `medium` for models before `gpt-5.1`.
- Structured outputs are supported in the Python SDK via `client.responses.parse(..., text_format=YourPydanticModel)`.
- Parsed output is returned on `response.output_parsed`.

Repo mapping:

- `emoji_bench/provider_eval.py` uses `client.responses.parse(...)`.
- `emoji_bench/model_registry.py` stores the configured default reasoning effort.
- `emoji_bench/eval_cli.py` allows `--reasoning-effort` as an override for OpenAI models.
