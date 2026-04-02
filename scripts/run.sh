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
SHARDS_PER_MODEL="${SHARDS_PER_MODEL:-1}"

MODELS=(
  gpt-4.1-mini
  gpt-5.4
  gpt-5.4-mini
  gpt-5.4-nano
  claude-sonnet-4-6
  claude-haiku-4-5
  mistral-large-2512
  mistral-medium-2508
)

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
fi

resolve_positive_integer() {
  local raw_value="$1"
  local variable_name="$2"

  if [[ "${raw_value}" =~ ^[0-9]+$ ]] && [[ "${raw_value}" -ge 1 ]]; then
    echo "${raw_value}"
    return 0
  fi

  echo "${variable_name} must be a positive integer" >&2
  exit 1
}

resolve_parallelism() {
  local total_jobs="$1"

  if [[ "${MODEL_PARALLELISM}" == "all" ]]; then
    echo "${total_jobs}"
    return 0
  fi

  if [[ "${MODEL_PARALLELISM}" =~ ^[0-9]+$ ]] && [[ "${MODEL_PARALLELISM}" -ge 1 ]]; then
    echo "${MODEL_PARALLELISM}"
    return 0
  fi

  echo "MODEL_PARALLELISM must be a positive integer or 'all'" >&2
  exit 1
}

format_job_label() {
  local model="$1"
  local shard_index="$2"
  local num_shards="$3"

  if [[ "${num_shards}" -le 1 ]]; then
    echo "${model}"
    return 0
  fi

  echo "${model} [shard $((shard_index + 1))/${num_shards}]"
}

wait_for_jobs_to_exit() {
  local pid

  for pid in "$@"; do
    if kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

terminate_jobs() {
  local signal="${1:-TERM}"
  shift || true

  local pid
  for pid in "$@"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "-${signal}" "${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    fi
  done
}

handle_interrupt() {
  local signal="$1"

  if [[ "${INTERRUPTED:-0}" -eq 1 ]]; then
    exit 130
  fi

  INTERRUPTED=1

  echo
  echo "Received ${signal}; terminating running model jobs..." >&2

  terminate_jobs TERM "${PIDS[@]+"${PIDS[@]}"}"
  wait_for_jobs_to_exit "${PIDS[@]+"${PIDS[@]}"}"

  exit 130
}

run_model() {
  local model="$1"
  local shard_index="$2"
  local num_shards="$3"
  local job_label
  job_label="$(format_job_label "${model}" "${shard_index}" "${num_shards}")"

  echo
  echo "Running model: ${job_label}"

  cmd=(
    uv run --extra openai --extra anthropic
    python scripts/evaluate_model.py
    "${DATASET_DIR}"
    --model "${model}"
    --num-shards "${num_shards}"
    --shard-index "${shard_index}"
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

  exec "${cmd[@]}"
}

wait_for_slot() {
  local max_parallel="$1"
  while true; do
    local active=0
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
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
  local next_labels=()

  for idx in "${!PIDS[@]}"; do
    local pid="${PIDS[$idx]}"
    local job_label="${PID_LABELS[$idx]}"

    if kill -0 "${pid}" 2>/dev/null; then
      next_pids+=("${pid}")
      next_labels+=("${job_label}")
      continue
    fi

    if wait "${pid}"; then
      echo "Completed model: ${job_label}"
    else
      echo "Model failed: ${job_label}" >&2
      terminate_jobs TERM "${next_pids[@]+"${next_pids[@]}"}"
      exit 1
    fi
  done

  if [[ "${#next_pids[@]}" -gt 0 ]]; then
    PIDS=("${next_pids[@]}")
    PID_LABELS=("${next_labels[@]}")
  else
    PIDS=()
    PID_LABELS=()
  fi
}

SHARDS_PER_MODEL_VALUE="$(resolve_positive_integer "${SHARDS_PER_MODEL}" "SHARDS_PER_MODEL")"
TOTAL_JOBS=$(( ${#MODELS[@]} * SHARDS_PER_MODEL_VALUE ))
MAX_PARALLEL="$(resolve_parallelism "${TOTAL_JOBS}")"
PIDS=()
PID_LABELS=()
INTERRUPTED=0

trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM

echo "Downloading dataset ${DATASET_REPO_ID} into ${DATASET_DIR}"
uv run --extra hf python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${DATASET_REPO_ID}",
    repo_type="dataset",
    local_dir="${DATASET_DIR}",
)
PY

echo "Shards per model: ${SHARDS_PER_MODEL_VALUE}"
echo "Concurrent eval jobs: ${MAX_PARALLEL}"

for model in "${MODELS[@]}"; do
  for (( shard_index=0; shard_index<SHARDS_PER_MODEL_VALUE; shard_index++ )); do
    local_job_label="$(format_job_label "${model}" "${shard_index}" "${SHARDS_PER_MODEL_VALUE}")"
    wait_for_slot "${MAX_PARALLEL}"
    run_model "${model}" "${shard_index}" "${SHARDS_PER_MODEL_VALUE}" &
    PIDS+=("$!")
    PID_LABELS+=("${local_job_label}")
  done
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
