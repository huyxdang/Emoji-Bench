# OpenAI Model Notes

Official sources checked on April 2, 2026:

- GPT-5.4 model page: `https://developers.openai.com/api/docs/models/gpt-5.4`
- GPT-5.4 mini model page: `https://developers.openai.com/api/docs/models/gpt-5.4-mini`
- GPT-5.4 nano model page: `https://developers.openai.com/api/docs/models/gpt-5.4-nano`
- All models index: `https://developers.openai.com/api/docs/models/all`

Key details used by Emoji-Bench:

- `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano` are available on the Responses API.
- OpenAI docs list a `128,000` max output token limit for the GPT-5.4 family pages used for this repo update.
- GPT-5.4 family models are reasoning models, so evaluator config should include a reasoning policy, not just a raw model ID.

Repo defaults:

- Emoji-Bench sets `default_max_output_tokens=50` for evaluation because the output schema is only `{has_error, error_step}`.
- Emoji-Bench sets `reasoning.effort="medium"` for the GPT-5.4 family unless overridden on the CLI.
