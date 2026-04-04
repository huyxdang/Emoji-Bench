#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.eval_cli import main


if __name__ == "__main__":
    main(
        description="Run a configured Gemini model on Emoji-Bench prompts and score has_error / error_step.",
        allowed_providers=("gemini",),
        default_model="gemini-3-flash-preview",
    )
