# Anthropic Model Notes

Official source checked on April 2, 2026:

- Models overview: `https://platform.claude.com/docs/en/about-claude/models/overview`
- Structured outputs compatibility: `https://platform.claude.com/docs/en/build-with-claude/structured-outputs`

Key details used by Emoji-Bench:

- Anthropic docs list `claude-sonnet-4-6` as the Claude Sonnet 4.6 Anthropic API alias.
- Anthropic docs list `claude-haiku-4-5` as the Claude Haiku 4.5 Anthropic API alias.
- Anthropic docs list `claude-haiku-4-5-20251001` as the Claude Haiku 4.5 snapshot ID.
- The current Anthropic docs list a `64k` max output token limit for Claude Sonnet 4.6 and Claude Haiku 4.5.
- Anthropic structured outputs are generally available for Claude Opus 4.6, Claude Sonnet 4.6, Claude Sonnet 4.5, Claude Opus 4.5, and Claude Haiku 4.5.
- Claude Sonnet 4.6 supports extended thinking, and Emoji-Bench now exposes both a baseline alias and a separate `claude-sonnet-4-6-reasoning` config.

Repo defaults:

- Emoji-Bench sets `default_max_output_tokens=50` for evaluation because the output schema is only `{has_error, error_step}`.
- Extended thinking is left disabled by default even though the configured models support it.
- The reasoning-specific Sonnet 4.6 config uses explicit extended thinking with the minimum 1024-token budget so it remains a distinct, labeled comparison point.
