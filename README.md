<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

Emoji-Bench asks a simple question:

**When a model makes an error in a reasoning chain, is it able to detect that error accurately?**

The goal is to test metacognitive error detection, not just problem solving. A model might be able to compute the right answer from scratch and still fail to notice that a shown derivation contains a bad step. Emoji-Bench is built to isolate that distinction.

This repo is now focused on a single benchmark condition: **`E-RECONV`**.

## What is `E-RECONV`?

In `E-RECONV`, a reasoning derivation contains exactly one invalid step, but the chain still reaches the correct final answer.

That matters because a "shallow" checker can often get away with re-executing the reasoning and comparing the endpoint:

- if the endpoint is wrong, say "there is an error"
- if the endpoint is right, say "no error"

This means the LLM isn't actually checking its steps - Rather, it's recomputing the whole reasoning chain, and comparing the final answers. This is not the behaviour we want, as we'd like to test whether the model can verify errors against rules and constraints. 

`E-RECONV` breaks that shortcut. The endpoint is still correct. The only way to succeed is to verify the steps themselves.

This makes `E-RECONV` the cleanest setting for the benchmark’s core question: does the model actually notice its own reasoning error, or does it accept an invalid derivation because the conclusion looks fine?

## Why Emojis? 
We use emojis as mathematical abstracts to prevent the model from being able to use its mathematical knowledge from its math training data. A model may know `1 + 1 = 2`, but it wouldn't know `🌸 (+) 🤗 = 👋`.

There are many substitutes to emojis. We could have also defined an operation like `1 + 7 = 4` - Our experiments demonstrate that results with either emojis or numbers for abstract operations yield similar results. The choice to keep it as emojis was partly to enhance the visuality of the project.

## Benchmark Shape

Each example gives the model:

- a procedurally generated **novel** formal system built from emoji symbols
- operation tables and optional transforms / derived operations
- a worked derivation
- a task: return `has_error` and `error_step`

Difficulty scales by system complexity:

| Difficulty | Symbols | Base Ops | Derived Ops | Transforms |
|---|---:|---:|---:|---:|
| Easy | 3 | 1 | 0 | 0 |
| Medium | 4 | 1 | 1 | 1 |
| Hard | 5 | 2 | 1 | 1 |
| Expert | 6 | 2 | 2 | 2 |

## Quick Start

Install the project and dev dependencies:

```bash
uv pip install -e ".[dev]"
```

If you want to evaluate OpenAI or Anthropic models through the shared CLI:

```bash
uv pip install -e ".[openai,anthropic]"
```

Set whichever API keys you need:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`

Run tests:

```bash
pytest
```

## Dataset

### Download The `E-RECONV` Dataset

```bash
uv run --extra hf python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="huyxdang/emoji-bench-e-reconv-1000",
    repo_type="dataset",
    local_dir="artifacts/emoji-bench-e-reconv-1000",
)
PY
```

### Generate An `E-RECONV` Dataset Locally

```bash
python scripts/generate_reconvergent_dataset.py \
  --dataset-name emoji-bench-e-reconv-1000 \
  --output-dir artifacts/emoji-bench-e-reconv-1000 \
  --count 1000
```

The generator guarantees the requested count by continuing to search until it finds enough reconvergent examples.

If you want a different dataset, change `--count` and optionally `--master-seed`.

## Preview Examples

To inspect the dataset in a more readable format:

```bash
python scripts/preview_dataset.py artifacts/emoji-bench-e-reconv-1000 --count 5
```

## Evaluate Models

List configured models:

```bash
python scripts/evaluate_model.py --list-models
```

Run a quick smoke test:

```bash
uv run --extra openai --extra anthropic python scripts/evaluate_model.py \
  artifacts/emoji-bench-e-reconv-1000 \
  --model gpt-5.4-mini \
  --limit 2
```

Run a larger reconvergent batch with the dedicated wrapper:

```bash
LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=8 ./scripts/run_reconv.sh
```

Run only one model:

```bash
LIMIT=10 SHARDS_PER_MODEL=2 MODEL_PARALLELISM=2 ./scripts/run_reconv.sh claude-sonnet-4-6-reasoning
```

Useful notes:

- model configs live in `emoji_bench/model_registry.py`
- provider request logic lives in `emoji_bench/provider_eval.py`
- reruns resume from existing `predictions.jsonl` unless you pass `--no-resume`
- eval outputs are written under `artifacts/evals/...`

## Reports

Aggregate eval outputs into CSV and HTML reports:

```bash
python scripts/analyze_evals.py artifacts/evals
```

For reconvergent-only runs, the wrapper already writes a scoped report:

```bash
./scripts/run_reconv.sh
```

By default that report is written to:

```text
artifacts/eval-report-e-reconv-1000
```

## Repo Map

- `emoji_bench/reconvergent_error_injector.py`: `E-RECONV` injection logic
- `emoji_bench/reconvergent_dataset.py`: exact-count reconvergent dataset generation
- `emoji_bench/generator.py`: formal-system generation
- `emoji_bench/chain_generator.py`: derivation generation
- `emoji_bench/benchmark.py`: benchmark-instance assembly
- `emoji_bench/provider_eval.py`: provider-specific evaluation requests
- `emoji_bench/reporting.py`: eval aggregation and report rendering
- `scripts/generate_reconvergent_dataset.py`: `E-RECONV` dataset generation
- `scripts/preview_dataset.py`: readable dataset preview
- `scripts/run_reconv.sh`: reconvergent evaluation wrapper

## Why Only `E-RECONV`

Earlier versions of the project included other error types as well (as you can tell from the extra files), and we did run experiments on them. But the project now centers only `E-RECONV`.

It is the cleanest experiment for the goal above. If the final answer is still correct, endpoint checking is no longer enough. A model either catches the bad step or it doesn’t. That makes `E-RECONV` the most direct test of whether models can accurately detect their own reasoning errors.