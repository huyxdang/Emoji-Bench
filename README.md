<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

Emoji-Bench is a benchmark for step-level error detection in procedurally generated formal systems built from emoji symbols. The point is to test whether a model actually verifies rules and intermediate steps, rather than just noticing that the final answer is wrong.

Each example gives the model:

- a random formal system with symbols, operation tables, transforms, and optional derived operations
- a worked derivation
- a task: return `has_error` and `error_step`

## What It Tests

The benchmark is designed to separate endpoint checking from real step verification.

| Difficulty | Symbols | Base Ops | Derived Ops | Transforms |
|---|---:|---:|---:|---:|
| Easy | 3 | 1 | 0 | 0 |
| Medium | 4 | 1 | 1 | 1 |
| Hard | 5 | 2 | 1 | 1 |
| Expert | 6 | 2 | 2 | 2 |

## Error Types

- `E-RES`: a valid rule is cited, but the result is wrong
- `E-INV`: the step cites a plausible-looking rule that does not exist
- `E-CASC`: one early step is wrong and the suffix is recomputed, so later steps are locally valid but the final answer is wrong
- `E-RECONV`: one early step is wrong and the suffix is recomputed, but the chain still reaches the correct final answer

`E-RECONV` is the most important anti-shortcut condition here. A model that only re-executes the chain and compares the endpoint can miss it completely.

## Quick Start

Install the base project and test dependencies:

```bash
uv pip install -e ".[dev]"
```

If you want to evaluate OpenAI or Anthropic models through the shared CLI, install those extras too:

```bash
uv pip install -e ".[openai,anthropic]"
```

Gemini and Mistral use direct HTTPS integrations in this repo, so they do not need extra Python packages. Set the relevant API key before running evaluation:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`

Run the test suite:

```bash
pytest
```

## Datasets

### Download The Mixed 2,000-Example Dataset

```bash
uv run --extra hf python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="huyxdang/emoji-bench-mixed-2000",
    repo_type="dataset",
    local_dir="artifacts/emoji-bench-mixed-2000",
)
PY
```

The evaluator can then point at `artifacts/emoji-bench-mixed-2000` directly.

### Generate A Mixed Dataset Locally

```bash
python scripts/generate_dataset.py \
  --dataset-name emoji-bench-mixed-2000 \
  --output-dir artifacts/emoji-bench-mixed-2000 \
  --count 2000 \
  --train-ratio 0 \
  --validation-ratio 0
```

### Generate An `E-RECONV`-Only Dataset With An Exact Count

```bash
python scripts/generate_reconvergent_dataset.py \
  --dataset-name emoji-bench-e-reconv-500 \
  --output-dir artifacts/emoji-bench-e-reconv-500 \
  --count 500
```

That script generates only `E-RECONV` rows and keeps searching until it reaches the requested count.

## Evaluate Models

List configured models:

```bash
python scripts/evaluate_model.py --list-models
```

Run one smoke test with the shared evaluator:

```bash
uv run --extra openai --extra anthropic python scripts/evaluate_model.py \
  artifacts/emoji-bench-mixed-2000 \
  --model gpt-5.4-mini \
  --limit 2
```

Run a Gemini model:

```bash
python scripts/evaluate_gemini.py \
  artifacts/emoji-bench-mixed-2000 \
  --model gemini-3-flash-preview \
  --limit 2
```

Run a larger batch with the wrapper:

```bash
LIMIT=500 ./scripts/run.sh
```

Run sharded full-dataset evals for the Gemini models:

```bash
LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=8 ./scripts/run.sh \
  gemini-3-flash-preview gemini-3.1-pro-preview
```

Useful notes:

- configured models live in `emoji_bench/model_registry.py`
- provider request logic lives in `emoji_bench/provider_eval.py`
- reruns resume from existing `predictions.jsonl` unless you pass `--no-resume`
- evaluation outputs are written under `artifacts/evals/...`

## Reports

Aggregate eval outputs into CSV and HTML reports:

```bash
python scripts/analyze_evals.py artifacts/evals
```

By default this writes report artifacts to `artifacts/eval-report`.

## Repo Map

- `emoji_bench/generator.py`: formal-system generation
- `emoji_bench/chain_generator.py`: step-by-step derivation generation
- `emoji_bench/error_injector.py`: `E-RES`, `E-INV`, and `E-CASC`
- `emoji_bench/reconvergent_error_injector.py`: `E-RECONV`
- `emoji_bench/benchmark.py`: benchmark-instance assembly
- `emoji_bench/provider_eval.py`: provider-specific evaluation requests
- `emoji_bench/reporting.py`: eval aggregation and report rendering
- `scripts/generate_dataset.py`: mixed dataset generation
- `scripts/generate_reconvergent_dataset.py`: exact-count `E-RECONV` dataset generation
- `scripts/evaluate_model.py`: shared model evaluator
- `scripts/run.sh`: multi-model evaluation wrapper

LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=4 ./scripts/run_reconv.sh claude-sonnet-4-6-reasoning