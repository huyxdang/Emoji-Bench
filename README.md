<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

**A benchmark for testing error detection in large language models using procedurally generated formal systems.**

Emoji-Bench measures whether LLMs perform genuine constraint evaluation or merely pattern-match on surface features when detecting errors in reasoning chains. It does this by constructing novel algebraic systems from emoji symbols -- systems that don't exist in any training corpus -- and asking models to verify multi-step derivations within them.

If a model can detect rule violations in a system it has never seen before, it must be performing real constraint checking. Pattern-matching from training data cannot help.

## Quick Start

```bash
uv pip install -e ".[dev]"
```

For model evaluation clients:

```bash
uv pip install -e ".[openai,anthropic]"
```

```python
from emoji_bench.generator import generate_system
from emoji_bench.formatter import format_system_for_prompt

system = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77)
print(system)
# FormalSystem('Cael System', seed=77, symbols={...}, ops=[⊕, ⊗], transforms=[inv])

print(format_system_for_prompt(system))
```

## How It Works

Each benchmark instance is a **formal system** -- a set of emoji symbols, operations defined by explicit tables, and transformation rules -- paired with a multi-step derivation chain that may contain an injected error. The model must verify each step against the rules and identify any violations.

### Difficulty Levels

| Level | Symbols | Base Ops | Derived Ops | Transforms |
|-------|---------|----------|-------------|------------|
| Easy | 3 | 1 | 0 | 0 |
| Medium | 4 | 1 | 1 | 1 |
| Hard | 5 | 2 | 1 | 1 |
| Expert | 6 | 2 | 2 | 2 |

### Example: Medium `E-CASC` Instance

The example below shows a medium-difficulty problem with one injected cascading error. Step 2 is wrong: `inv(🍄)` should reduce to `🧄`, but the chain says `🍄`. Steps 3-5 are then locally valid given that wrong intermediate value, which is exactly what makes `E-CASC` interesting.

```text
Symbols: {🪸, 🪩, 🧄, 🍄}

Operation ⊕ (defined by table):

| ⊕ | 🪸 | 🪩 | 🧄 | 🍄 |
|---|---|---|---|---|
| **🪸** | 🧄 | 🍄 | 🪩 | 🧄 |
| **🪩** | 🧄 | 🍄 | 🍄 | 🪸 |
| **🧄** | 🍄 | 🪩 | 🪸 | 🪸 |
| **🍄** | 🪸 | 🧄 | 🪩 | 🪩 |

Derived operation ⊗:
x ⊗ y = (x ⊕ x) ⊕ y

Transformation "inv":
  inv(🪸) = 🪩
  inv(🪩) = 🪸
  inv(🧄) = 🍄
  inv(🍄) = 🧄
  Distribution property: inv(x ⊕ y) = inv(x) ⊕ inv(y)

Start: (inv(🧄) ⊗ inv(🍄))

Step 1: (inv(🧄) ⊗ inv(🍄)) = (🍄 ⊗ inv(🍄))    [by inv]
Step 2: (🍄 ⊗ inv(🍄)) = (🍄 ⊗ 🍄)    [by inv]
Step 3: (🍄 ⊗ 🍄) = ((🍄 ⊕ 🍄) ⊕ 🍄)    [by definition of ⊗]
Step 4: ((🍄 ⊕ 🍄) ⊕ 🍄) = (🪩 ⊕ 🍄)    [by ⊕ table]
Step 5: (🪩 ⊕ 🍄) = 🪸    [by ⊕ table]

Result: 🪸
```

### Error Types

Emoji-Bench currently uses three implemented error types and tracks two additional deferred types.

- `E-RES` -- **Wrong result.** The cited rule is real and the operands are the right ones, but the output symbol is wrong. In the current prompt format, this is only injected on the final step; changing an earlier full-expression rewrite would otherwise create an obvious continuity break in the later steps.
- `E-INV` -- **Invented rule.** The step cites a plausible-looking rule that is not actually defined in the formal system, such as a nonexistent transform or operator definition.
- `E-CASC` -- **Cascading wrong result.** One earlier step is made wrong, and the remaining suffix is recomputed so that later steps are locally valid given the wrong intermediate value. This produces a fluent-looking derivation with a single root violation.
- `E-OP` -- **Wrong operands.** Deferred for now. This would mean the model must detect that the rule was applied to the wrong subexpression or wrong pair of inputs, even if the claimed rewrite looks superficially plausible.
- `E-SUB` -- **Subtle wrong lookup.** Deferred for now. This would be a near-miss table lookup where the returned symbol is close to the correct one, making the mistake harder to catch than a random wrong result.

### Status / TODO

Currently supported error types are `E-RES`, `E-INV`, and `E-CASC`. `E-OP` and `E-SUB` are intentionally deferred for now because the current prompt format exposes full-expression rewrites rather than explicit local reductions.

- [x] Procedural generation of formal systems across Easy / Medium / Hard / Expert settings
- [x] Recursive expression interpreter for base ops, derived ops, and transformations
- [x] Deterministic derivation-chain generation with reproducible seeds
- [x] Prompt generation for clean and error-injected benchmark instances
- [x] Error injection for `E-RES`, `E-INV`, and `E-CASC`
- [x] JSON round-tripping for generated formal systems
- [x] End-to-end test coverage across generator, formatter, interpreter, chain builder, and benchmark APIs
- [x] Evaluation runners for configured OpenAI and Anthropic models
- [ ] Suspicious-but-correct benchmark condition
- [ ] Rule-visibility ablation / no-rules control
- [ ] Familiar-domain arithmetic mirror condition
- [ ] Local-reduction prompt mode to support deferred `E-OP` and `E-SUB`

---

## Core Components

### 1. Symbol Sampling (`symbols.py`)

A curated pool of ~30 emoji chosen for three properties: they tokenize as a single token across major model families, they carry no mathematical or logical associations, and they are visually distinct at small sizes. Each system samples a random subset, and the randomized assignment prevents models from learning fixed symbol-role mappings across instances.

### 2. Operation Tables (`operations.py`)

Each base operation is defined by an explicit n x n lookup table mapping every pair of symbols to a result symbol. There are two generation strategies:

- **Random magma tables** -- used when no transformations are needed (Easy difficulty). Every cell is filled independently at random. These tables have no algebraic structure whatsoever, maximizing novelty.
- **Automorphism-compatible tables** -- used when transformations are needed (Medium through Expert). Rather than generating a random table and hoping it has automorphisms (which almost never works -- most random binary operations have trivial automorphism groups), we flip the problem: pick the desired automorphism first, then build the table to be compatible with it. This is done by grouping all (a, b) input pairs into orbits under the chosen permutation, assigning one random result per orbit, and propagating via the constraint `sigma(a op b) = sigma(a) op sigma(b)`. Since a single permutation generates a cyclic group, each orbit is a simple cycle, making propagation conflict-free by construction.

### 3. Transformations (`transforms.py`)

A transformation is a permutation of the symbol set that "distributes" over the operations: `t(x op y) = t(x) op t(y)` for all x, y. In algebraic terms, this is an automorphism. The generator picks a random permutation sigma of sufficient order (>= n_transformations + 1), then uses the powers sigma, sigma^2, ..., sigma^k as the transforms. The `validate_distribution_property` function performs an exhaustive n^2 check to verify correctness.

### 4. Derived Operations (`expressions.py`, `interpreter.py`)

Derived operations add a layer of indirection: they are defined in terms of base operations using templates like `x * y = (x + y) + x`. The model must understand the definition and expand it correctly to verify steps involving derived operations. Three templates are available:

- `compose_left`: `x op2 y = (x op1 y) op1 x`
- `inv_compose`: `x op2 y = inv(x op1 y)`
- `double_left`: `x op2 y = (x op1 x) op1 y`

The interpreter evaluates these by template expansion at evaluation time, not by rewriting the AST. This keeps the expression tree clean and matches what the model sees in the prompt.

### 5. Expression AST & Interpreter (`expressions.py`, `interpreter.py`)

Expressions are represented as a simple recursive tree: `SymbolLiteral | BinaryOp | UnaryTransform`. The interpreter walks the tree with `match/case`, resolving base operations via table lookup, derived operations via template expansion, and transformations via the explicit mapping. This same interpreter is used for chain generation (verifying correct steps) and error detection (computing what the result *should* have been).

### 6. Prompt Formatting (`formatter.py`)

The formatter renders a `FormalSystem` into the Markdown format that gets presented to the model under test. Operation tables become Markdown tables, derived operations are shown in algebraic notation with the actual operator symbols (not internal names), and transformations list their explicit mappings plus the distribution property statement. All internal identifiers (like `op0`) are resolved to their display symbols (like `⊕`) so the prompt reads naturally.

### 7. System Generation (`generator.py`)

The top-level orchestrator ties everything together. Given parameters (n_symbols, n_base_ops, n_derived_ops, n_transformations, random_seed), it:

1. Samples symbols from the curated pool
2. Generates a random system name
3. Builds base operation tables (random magma or automorphism-compatible, depending on whether transforms are needed)
4. Selects transformations as powers of a chosen permutation
5. Picks derived operation templates (filtering out `inv_compose` if no transforms exist)
6. Runs a full consistency validation pass

The seed makes everything deterministic and reproducible.

---

## Running Tests

```bash
pytest tests/ -v
```

## Evaluating Models

Emoji-Bench now includes a shared multi-provider evaluator plus provider-specific wrappers:

```bash
python scripts/evaluate_model.py --list-models
python scripts/evaluate_openai.py artifacts/emoji-bench-mixed-2000 --model gpt-5.4-mini
python scripts/evaluate_anthropic.py artifacts/emoji-bench-mixed-2000 --model claude-sonnet-4-6
python scripts/evaluate_model.py artifacts/emoji-bench-mixed-2000 --model mistral-medium-2508
```

Configured models currently include:

- `gpt-4.1-mini` (non-reasoning baseline)
- `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` with `reasoning.effort=medium`
- `claude-sonnet-4-6`, `claude-haiku-4-5`
- `mistral-large-2512`, `mistral-medium-2508`

The model registry lives in `emoji_bench/model_registry.py`. The shared CLI is `scripts/evaluate_model.py`, and the provider-specific request handling lives in `emoji_bench/provider_eval.py`.

Current evaluator defaults now favor reasoning headroom over minimum-cost runs:

- `default_max_output_tokens=512` for most configured models
- `gpt-5.4-nano` defaults to `2048` because `512` was too small for reliable structured-output completion on Emoji-Bench
- GPT-5.4 family models use medium reasoning by default
- Anthropic extended thinking is supported by the configured Claude models but disabled by default
- Mistral models use the same default token budget unless you explicitly pass `--max-output-tokens`

As of April 2, 2026, the config values above were checked against official OpenAI and Anthropic docs while adding this integration layer.

### Download The Evaluation Dataset

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

After download, the evaluator can point at the directory directly; it will read `test.jsonl` from there.

### Run One Model

GPT-5.4 full-dataset example:

```bash
uv run --extra openai --extra anthropic python scripts/evaluate_model.py \
  artifacts/emoji-bench-mixed-2000 \
  --model gpt-5.4
```

Add `--limit 2` if you just want a smoke test.

GPT-5.4 mini example:

```bash
uv run --extra openai --extra anthropic python scripts/evaluate_model.py \
  artifacts/emoji-bench-mixed-2000 \
  --model gpt-5.4-mini \
  --limit 2
```

Mistral example:

```bash
.venv/bin/python scripts/evaluate_model.py \
  artifacts/emoji-bench-mixed-2000 \
  --model mistral-medium-2508 \
  --limit 2
```

Useful overrides:

- raise token budget: `--max-output-tokens 1024`
- increase GPT-5.x reasoning: `--reasoning-effort high`
- resume-safe repeated runs: the evaluator skips examples already present in `predictions.jsonl`
- if you do not pass `--max-output-tokens`, the evaluator uses `default_max_output_tokens` from `emoji_bench/model_registry.py`

### Run All Models

Use the wrapper at [`scripts/run.sh`](scripts/run.sh).

Smoke test on all configured models:

```bash
./scripts/run.sh
```

From another terminal, gracefully stop the current running `bash` process for the wrapper:

```bash
pkill -TERM -f 'scripts/run\.sh'
```

Smoke test only the four current expansion targets:

```bash
./scripts/run.sh gpt-5.4-nano claude-sonnet-4-6 mistral-large-2512 mistral-medium-2508
```

Run the first 500 examples instead of the full 2,000-example test split:

```bash
LIMIT=500 ./scripts/run.sh
```

Run the full dataset:

```bash
LIMIT=all ./scripts/run.sh
```

Run only a subset of models:

```bash
LIMIT=500 ./scripts/run.sh gpt-5.4-mini claude-sonnet-4-6
```

Run the subset discussed here on the full dataset:

```bash
LIMIT=all ./scripts/run.sh gpt-5.4-nano claude-sonnet-4-6 mistral-large-2512 mistral-medium-2508
```

### Parallelism

By default, [`scripts/run.sh`](scripts/run.sh) runs models sequentially. You can parallelize across models by setting `MODEL_PARALLELISM`.

Run two model processes at a time:

```bash
LIMIT=500 MODEL_PARALLELISM=2 ./scripts/run.sh
```

Run all configured models simultaneously:

```bash
LIMIT=500 MODEL_PARALLELISM=all ./scripts/run.sh
```

Why not always run all six at once? Because the bottleneck is usually provider-side rate limits, quota, or burst capacity rather than local CPU. Running everything simultaneously is supported, but it can lead to more API throttling, longer retries, or uneven completion times across providers.

### Per-Model Sharding

You can also split each model run into multiple stable shards by setting `SHARDS_PER_MODEL`. Each shard writes to its own directory under the model output directory, and the reporting step aggregates them automatically.

Run four shards per model with up to eight shard jobs active at once:

```bash
LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=8 ./scripts/run.sh
```

Run four shards for the OpenAI / Anthropic / Mistral subset:

```bash
LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=8 ./scripts/run.sh \
  gpt-5.4-nano claude-sonnet-4-6 mistral-large-2512 mistral-medium-2508
```

If you use `MODEL_PARALLELISM=all`, it now means all model-shard jobs, not just all models:

```bash
LIMIT=all SHARDS_PER_MODEL=4 MODEL_PARALLELISM=all ./scripts/run.sh
```

For manual runs, pass the shard count and zero-based shard index directly:

```bash
uv run --extra openai --extra anthropic python scripts/evaluate_model.py \
  artifacts/emoji-bench-mixed-2000 \
  --model gpt-5.4-mini \
  --num-shards 4 \
  --shard-index 0
```

### Reports And Analysis

Every evaluator run writes per-model outputs under `artifacts/evals/...`. The reporting script aggregates those runs into machine-readable summaries plus an HTML dashboard.

Run `scripts/analyze_evals.py` from the repo root to build a combined report. By default it writes to `artifacts/eval-report`:

```bash
python3 scripts/analyze_evals.py artifacts/evals
```

Pass `--output-dir` if you want the report written somewhere else.

The report includes:

- detection metrics: accuracy, precision, recall, F1, clean false-positive rate, error false-negative rate
- localization metrics: joint accuracy, exact step accuracy on error rows, step accuracy when an error was detected, mean absolute step distance, within-one-step rate, off-by-one rate
- slice analysis: by model, difficulty, error type, expected error step, and actual step count
- operational metrics: latency and token usage when the provider returns them

Artifacts:

- `summary.json`
- `by_model.csv`
- `by_model_difficulty.csv`
- `by_model_error_type.csv`
- `by_model_expected_step.csv`
- `by_model_actual_step_count.csv`
- `report.html`

140 tests covering unit tests for each module, integration tests at all four difficulty levels, benchmark error-injection paths, evaluator/model-config coverage, and a 50-seed stress test verifying consistency across generated systems.

## Project Structure

```
emoji_bench/
    benchmark.py      # Benchmark-instance generation
    benchmark_types.py # Benchmark enums and metadata
    chain_generator.py # Step-by-step derivation generation
    chain_types.py     # ChainStep / DerivationChain dataclasses
    error_injector.py  # Error-type injection logic
    types.py          # Core dataclasses: Symbol, OperationTable, FormalSystem, etc.
    symbols.py        # Emoji pool and sampling
    operations.py     # Table generation (random magma + automorphism-compatible)
    expressions.py    # Expression AST and rendering
    interpreter.py    # Recursive expression evaluator
    transforms.py     # Automorphism search and validation
    generator.py      # Top-level system generator
    formatter.py      # JSON serialization and prompt formatting
    prompt_formatter.py # Full benchmark prompt rendering
tests/
    test_*.py         # 140 tests
```


### Initial Results

Results on the full 2,000-example `emoji-bench-mixed-2000` dataset with `gpt-4.1-mini` (non-reasoning baseline):

| Metric | Overall | Easy | Medium | Hard | Expert |
|--------|---------|------|--------|------|--------|
| Detection accuracy | 67.5% | 74.1% | 69.7% | 62.2% | 63.8% |
| Joint accuracy | 49.2% | 49.9% | 51.1% | 48.6% | 47.4% |
| Localization (error rows) | 50.9% | 59.4% | 51.1% | 46.1% | 46.9% |
| False positive rate (clean) | 55.6% | 78.4% | 48.8% | 44.0% | 51.2% |

By error type:

| Error Type | Detection | Joint Accuracy | Localization |
|------------|-----------|----------------|--------------|
| Clean | 44.4% | 44.4% | -- |
| E-RES | 77.2% | 40.0% | 51.8% |
| E-INV | 75.6% | 50.0% | 66.1% |
| E-CASC | 72.7% | 62.7% | 86.2% |

Joint accuracy is essentially flat across difficulty levels (~47-51%), suggesting `gpt-4.1-mini` is near its floor on this task regardless of problem complexity. Stronger models may show more separation.

README should be like: 
- What this is about 
- How to run this