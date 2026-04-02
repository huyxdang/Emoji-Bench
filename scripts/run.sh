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
MODEL_PARALLELISM="${MODEL_PARALLELISM:-1}"

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

resolve_parallelism() {
  if [[ "${MODEL_PARALLELISM}" == "all" ]]; then
    echo "${#MODELS[@]}"
    return 0
  fi

  if [[ "${MODEL_PARALLELISM}" =~ ^[0-9]+$ ]] && [[ "${MODEL_PARALLELISM}" -ge 1 ]]; then
    echo "${MODEL_PARALLELISM}"
    return 0
  fi

  echo "MODEL_PARALLELISM must be a positive integer or 'all'" >&2
  exit 1
}

run_model() {
  local model="$1"

  echo
  echo "Running model: ${model}"

  cmd=(
    uv run --extra openai --extra anthropic
    python scripts/evaluate_model.py
    "${DATASET_DIR}"
    --model "${model}"
    --request-delay-seconds "${REQUEST_DELAY_SECONDS}"
    --retry-delay-seconds "${RETRY_DELAY_SECONDS}"
    --max-retries "${MAX_RETRIES}"
  )

  if [[ -n "${LIMIT}" && "${LIMIT}" != "all" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  if [[ -n "${MAX_OUTPUT_TOKENS}" ]]; then
    cmd+=(--max-output-tokens "${MAX_OUTPUT_TOKENS}")
  fi

  "${cmd[@]}"
}

wait_for_slot() {
  local max_parallel="$1"
  while true; do
    local active=0
    for pid in "${PIDS[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        active=$((active + 1))
      fi
    done

    if [[ "${active}" -lt "${max_parallel}" ]]; then
      return 0
    fi

    collect_finished_jobs
    sleep 1
  done
}

collect_finished_jobs() {
  local next_pids=()
  local next_models=()

  for idx in "${!PIDS[@]}"; do
    local pid="${PIDS[$idx]}"
    local model="${PID_MODELS[$idx]}"

    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_models+=("${model}")
      continue
    fi

    if wait "${pid}"; then
      echo "Completed model: ${model}"
    else
      echo "Model failed: ${model}" >&2
      terminate_jobs "${next_pids[@]}"
      exit 1
    fi
  done

  PIDS=("${next_pids[@]:-}")
  PID_MODELS=("${next_models[@]:-}")
}

terminate_jobs() {
  for pid in "$@"; do
    kill "${pid}" 2>/dev/null || true
  done
}

MAX_PARALLEL="$(resolve_parallelism)"
PIDS=()
PID_MODELS=()

echo "Downloading dataset ${DATASET_REPO_ID} into ${DATASET_DIR}"
uv run --extra hf python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${DATASET_REPO_ID}",
    repo_type="dataset",
    local_dir="${DATASET_DIR}",
)
PY

echo "Model parallelism: ${MAX_PARALLEL}"

for model in "${MODELS[@]}"; do
  wait_for_slot "${MAX_PARALLEL}"
  run_model "${model}" &
  PIDS+=("$!")
  PID_MODELS+=("${model}")
done

while [[ "${#PIDS[@]}" -gt 0 ]]; do
  collect_finished_jobs
  sleep 1
done

echo
echo "Building combined report in ${REPORT_DIR}"
python3 scripts/analyze_evals.py artifacts/evals --output-dir "${REPORT_DIR}"

echo
echo "Done."
echo "Report: ${REPORT_DIR}/report.html"
echo "Summary: ${REPORT_DIR}/summary.json"
