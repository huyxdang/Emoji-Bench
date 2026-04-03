#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.metric_extract import (
    build_key_metric_payload,
    load_report_summary,
    write_key_metric_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Detection Rate, Joint Detection, and False Positives from "
            "Emoji-Bench report artifacts."
        ),
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="artifacts/eval-report",
        help=(
            "Path to an analyze_evals output directory or a summary.json file. "
            "Defaults to artifacts/eval-report."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where the key metric CSV/JSON exports will be written. "
            "Defaults to <report>/key-metrics."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report, summary_path = load_report_summary(args.report)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else summary_path.parent / "key-metrics"
    )
    payload = build_key_metric_payload(report, source_summary_path=summary_path)
    written_dir = write_key_metric_artifacts(payload, output_dir=output_dir)

    stdout_summary = {
        "source_summary": str(summary_path.resolve()),
        "output_dir": str(written_dir.resolve()),
        "overall": payload["overall"],
        "by_model_rows": len(payload["by_model"]),
        "by_model_difficulty_rows": len(payload["by_model_difficulty"]),
    }
    print(json.dumps(stdout_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
