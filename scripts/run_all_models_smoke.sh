#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-huyxdang/emoji-bench-mixed-2000}"
DATASET_DIR="${DATASET_DIR:-artifacts/emoji-bench-mixed-2000}"
REPORT_DIR="${REPORT_DIR:-artifacts/eval-report-smoke}"
LIMIT="${LIMIT:-2}"
REQUEST_DELAY_SECONDS="${REQUEST_DELAY_SECONDS:-0.2}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-2.0}"
MAX_RETRIES="${MAX_RETRIES:-3}"

MODELS=(
  gpt-4.1-mini
  gpt-5.4
  gpt-5.4-mini
  gpt-5.4-nano
  claude-sonnet-4-6
  claude-haiku-4-5
)

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
fi

echo "Downloading dataset ${DATASET_REPO_ID} into ${DATASET_DIR}"
uv run --extra hf python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${DATASET_REPO_ID}",
    repo_type="dataset",
    local_dir="${DATASET_DIR}",
)
PY

for model in "${MODELS[@]}"; do
  echo
  echo "Running model: ${model}"

  cmd=(
    uv run --extra openai --extra anthropic
    python scripts/evaluate_model.py
    "${DATASET_DIR}"
    --model "${model}"
    --limit "${LIMIT}"
    --request-delay-seconds "${REQUEST_DELAY_SECONDS}"
    --retry-delay-seconds "${RETRY_DELAY_SECONDS}"
    --max-retries "${MAX_RETRIES}"
  )

  if [[ -n "${MAX_OUTPUT_TOKENS}" ]]; then
    cmd+=(--max-output-tokens "${MAX_OUTPUT_TOKENS}")
  fi

  "${cmd[@]}"
done

echo
echo "Building combined report in ${REPORT_DIR}"
python3 scripts/analyze_evals.py artifacts/evals --output-dir "${REPORT_DIR}"

echo
echo "Done."
echo "Report: ${REPORT_DIR}/report.html"
echo "Summary: ${REPORT_DIR}/summary.json"
