<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

Emoji-Bench is a benchmark for step-level error verification in novel formal systems. It asks a narrow question with a high bar:

**When a model is shown a worked derivation containing an error, can it detect that an error exists and identify the exact first wrong step?**

This repository is now focused on a single benchmark condition: **`E-RECONV`**.

Links: [Hugging Face dataset](https://huggingface.co/datasets/huyxdang/emoji-bench-e-reconv-1000)

## TL;DR

- `E-RECONV` inserts exactly one invalid intermediate step while preserving the correct final answer.
- That blocks the cheap shortcut of re-solving the problem from scratch and comparing endpoints.
- The benchmark therefore tests **step-by-step verification**, not just problem solving.
- In current project results, reasoning-enabled models are much stronger at **localizing** the first bad step, not just saying that something is wrong.

## Why This Benchmark Exists

Many self-error-detection benchmarks are easier to game than they look.

- They rely on human annotations, which are expensive and hard to scale.
- They use familiar domains such as mathematics, where models can pattern-match from training data.
- They often use cascading errors, where one bad step causes a wrong final answer.

That last point matters. If the final answer is wrong, a shallow checker can often get away with:

- re-deriving the solution from scratch
- comparing the endpoint
- declaring "there is an error" without actually verifying where the chain broke

Emoji-Bench is designed to remove that shortcut.

- The tasks are procedurally generated.
- The formal systems are novel and absent from training corpora.
- The benchmark condition is reconvergent: the final answer is still correct even though one step is not.

## What Is `E-RECONV`?

In `E-RECONV`, a derivation contains exactly one invalid step, but the chain still reaches the correct final answer.

That means endpoint checking is useless. A model cannot succeed just by recomputing the answer and confirming that the result matches. It has to inspect the derivation itself and verify each step against the rules.

## Example

Below is a minimal reconvergent example in the same style used by the benchmark.

```text
=== RULES ===
Symbols: {🐙, 🪬, 🫑}
Operation ⊕ (defined by table):

      🐙  🪬  🫑
🐙    🐙  🐙  🪬
🪬    🐙  🪬  🫑
🫑    🐙  🪬  🪬

=== YOUR WORKING OUT ===
Start: ((🫑 ⊕ 🐙) ⊕ (🫑 ⊕ 🪬))
Step 1: (🫑 ⊕ 🐙) = 🐙
Step 2: (🫑 ⊕ 🪬) = 🐙
Step 3: (🐙 ⊕ 🐙) = 🐙
Result: 🐙

=== TASK ===
Check whether your working out contains an error. It may or may not contain an error.
```

The final answer is `🐙`, which is still correct. But `Step 2` is invalid: from the table, `🫑 ⊕ 🪬 = 🪬`, not `🐙`.

The intended structured output is:

```json
{"has_error": true, "error_step": 2}
```

## Benchmark Shape

Each example gives the model:

- a procedurally generated formal system built from emoji symbols
- operation tables and optional transforms or derived operations
- a worked derivation
- a task to return `has_error` and `error_step`

Difficulty scales by system complexity:

| Difficulty | Symbols | Base Ops | Derived Ops | Transforms |
|---|---:|---:|---:|---:|
| Easy | 3 | 1 | 0 | 0 |
| Medium | 4 | 1 | 1 | 1 |
| Hard | 5 | 2 | 1 | 1 |
| Expert | 6 | 2 | 2 | 2 |

## Metrics

Two metrics matter most:

- `Detection Rate`: does the model correctly identify that an error is present?
- `Error Localized`: does the model also name the exact first wrong step?

The gap between those two numbers is informative. A model can often tell that "something is wrong" without actually tracing the error to its source.

## Quick Start

### Requirements

- Python `>=3.11`
- `uv`

### Install

Install the package, test dependencies, and optional helpers in one step:

```bash
uv pip install -e ".[dev,hf,openai,anthropic]"
```

Provider notes:

- OpenAI models require `OPENAI_API_KEY`.
- Anthropic models require `ANTHROPIC_API_KEY`.
- Gemini models require `GEMINI_API_KEY`.
- Mistral models require `MISTRAL_API_KEY`.
- Gemini and Mistral requests use the standard library HTTP client, so there is no separate extra to install for them.

Run tests:

```bash
pytest
```

### Download The Public `E-RECONV` Dataset

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

Preview a few prompts:

```bash
python scripts/preview_dataset.py artifacts/emoji-bench-e-reconv-1000 --count 3
```

Run a small evaluation:

```bash
uv run --extra openai python scripts/evaluate_model.py \
  artifacts/emoji-bench-e-reconv-1000 \
  --model gpt-5.4-mini \
  --limit 2
```

Build reports:

```bash
python scripts/analyze_evals.py artifacts/evals
```

## Dataset

### Download

The public dataset lives at:

- `huyxdang/emoji-bench-e-reconv-1000`

It contains 1,000 prompt-only `E-RECONV` benchmark instances.

### Generate Locally

```bash
python scripts/generate_reconvergent_dataset.py \
  --dataset-name emoji-bench-e-reconv-1000 \
  --output-dir artifacts/emoji-bench-e-reconv-1000 \
  --count 1000
```

The generator keeps searching until it finds enough reconvergent examples to satisfy the exact requested count.

Useful knobs:

- change `--count` to generate a smaller or larger dataset
- change `--master-seed` for a different deterministic sample
- use `--target-length` or `--length-overrides` to adjust derivation length by difficulty

### File Layout

A generated dataset directory contains:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `manifest.json`
- `README.md`

### Key Fields

The main dataset fields are:

- `example_id`: unique row id
- `split`: `train`, `validation`, or `test`
- `difficulty`: `easy`, `medium`, `hard`, or `expert`
- `has_error`: whether the prompt contains an error
- `expected_error_step`: the ground-truth first wrong step
- `prompt`: the full prompt shown to the model
- `system_json`: serialized formal system for reproducibility
- `system_seed`, `chain_seed`, `error_seed`: generation metadata

## Evaluate Models

List configured models:

```bash
python scripts/evaluate_model.py --list-models
```

Run a smoke test on a local dataset directory or `test.jsonl` file:

```bash
python scripts/evaluate_model.py \
  artifacts/emoji-bench-e-reconv-1000 \
  --model gpt-5.4-mini \
  --limit 5
```

Run a larger reconvergent batch with the wrapper:

```bash
LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=8 ./scripts/run_reconv.sh
```

Run a single model:

```bash
LIMIT=10 SHARDS_PER_MODEL=2 MODEL_PARALLELISM=2 ./scripts/run_reconv.sh claude-sonnet-4-6-reasoning
```

Useful notes:

- model configs live in `emoji_bench/model_registry.py`
- provider request logic lives in `emoji_bench/provider_eval.py`
- reruns resume from existing `predictions.jsonl` unless you pass `--no-resume`
- default eval outputs are written under `artifacts/evals/...`
- the wrapper downloads the public dataset into `artifacts/emoji-bench-e-reconv-1000` by default

## Reports

Aggregate evaluation outputs into CSV and HTML reports:

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
- `emoji_bench/reporting.py`: evaluation aggregation and report rendering
- `scripts/generate_reconvergent_dataset.py`: `E-RECONV` dataset generation
- `scripts/preview_dataset.py`: terminal-friendly prompt preview
- `scripts/evaluate_model.py`: shared evaluation CLI
- `scripts/run_reconv.sh`: reconvergent evaluation wrapper

## Why Emojis?

Emoji symbols make the formal systems visually legible while keeping them novel. A model may know `1 + 1 = 2` from training, but it does not come pre-trained on a fresh symbolic system such as `🌸 (+) 🤗 = 👋`.

The benchmark could use other abstract labels as well. Earlier experiments with numeric relabelings produced similar behavior, but emojis make the tasks easier to inspect and discuss.

## Why Only `E-RECONV`?

Earlier versions of the project included multiple error types, and the repository still contains some of that machinery. The benchmark now centers on `E-RECONV` because it is the cleanest test of step-level verification.

If the final answer remains correct, endpoint comparison no longer helps. A model either catches the bad step or it does not.

## Contributing

Issues and pull requests are welcome, especially around:

- benchmark design
- provider integrations
- evaluation/reporting improvements
- analysis of model behavior on `E-RECONV`

## License

MIT. See `LICENSE`.
