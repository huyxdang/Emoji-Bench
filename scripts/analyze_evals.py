#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct `python scripts/...` execution from a repo checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emoji_bench.reporting import load_prediction_rows, summarize_prediction_rows, write_report_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Emoji-Bench evaluation outputs, compute comparison metrics, "
            "and write JSON/CSV/HTML report artifacts."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more predictions.jsonl files, eval output directories, or parent "
            "directories to scan recursively for predictions.jsonl files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval-report",
        help="Directory where summary.json, CSV tables, and report.html will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rows, input_paths = load_prediction_rows(args.paths)
    report = summarize_prediction_rows(rows)
    output_dir = write_report_artifacts(
        report,
        output_dir=args.output_dir,
        input_paths=input_paths,
    )

    stdout_summary = {
        "input_paths": input_paths,
        "output_dir": str(output_dir.resolve()),
        "total_rows": report["total_rows"],
        "models": report["models"],
        "overall": report["overall"],
    }
    print(json.dumps(stdout_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
