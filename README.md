<p align="center">
  <img src="public/emoji-bench.png" alt="Emoji-Bench" width="100%" />
</p>

# Emoji-Bench

**A benchmark for testing error detection in large language models using procedurally generated formal systems.**

Emoji-Bench measures whether LLMs perform genuine constraint evaluation or merely pattern-match on surface features when detecting errors in reasoning chains. It does this by constructing novel algebraic systems from emoji symbols -- systems that don't exist in any training corpus -- and asking models to verify multi-step derivations within them.

If a model can detect rule violations in a system it has never seen before, it must be performing real constraint checking. Pattern-matching from training data cannot help.

## Quick Start

```bash
pip install -e ".[dev]"
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

### Status / TODO

Currently supported error types are `E-RES`, `E-INV`, and `E-CASC`. `E-OP` and `E-SUB` are intentionally deferred for now because the current prompt format exposes full-expression rewrites rather than explicit local reductions.

- [x] Procedural generation of formal systems across Easy / Medium / Hard / Expert settings
- [x] Recursive expression interpreter for base ops, derived ops, and transformations
- [x] Deterministic derivation-chain generation with reproducible seeds
- [x] Prompt generation for clean and error-injected benchmark instances
- [x] Error injection for `E-RES`, `E-INV`, and `E-CASC`
- [x] JSON round-tripping for generated formal systems
- [x] End-to-end test coverage across generator, formatter, interpreter, chain builder, and benchmark APIs
- [ ] Suspicious-but-correct benchmark condition
- [ ] Rule-visibility ablation / no-rules control
- [ ] Familiar-domain arithmetic mirror condition
- [ ] Scoring / evaluation harness for model responses
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

128 tests covering unit tests for each module, integration tests at all four difficulty levels, benchmark error-injection paths, and a 50-seed stress test verifying consistency across 200 generated systems.

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
    test_*.py         # 128 tests
```


### Initial Results
─ Worked for 1m 03s ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Ran it on the Hugging Face dataset snapshot from huyxdang/emoji-bench-e-casc-200 with gpt-4.1-mini.

  Results on all 200 examples:

  - has_error_accuracy: 0.73
  By difficulty:

  - easy: 0.96 / 0.68 / 0.68
  - medium: 0.68 / 0.58 / 0.58
  - hard: 0.68 / 0.64 / 0.64
  - expert: 0.60 / 0.54 / 0.54