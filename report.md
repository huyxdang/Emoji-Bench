## Problem

A central open question in LLM metacognition is whether models genuinely detect errors or merely pattern-match on what mistakes "look like" in training data. Existing benchmarks — CorrectBench (Tie et al., 2025), Self-Correction Bench (Tsui, 2025), SelfCheckGPT (Manakul et al., 2023) — all operate in domains heavily represented in pretraining (arithmetic, algebra, code), making genuine constraint evaluation indistinguishable from memorized error-correction patterns.

We set out to measure genuine metacognitive reasoning in LLMs. 

## Solution

Emoji-Bench constructs **completely novel formal systems** — invented algebras built from emoji symbols (🦩 ⊕ 🧲 = 🪣) with procedurally generated operation tables, derived operations, and transformation rules. Models are given the rules and a multi-step derivation chain, then asked to verify whether each step correctly applies the rules. Some chains contain injected errors of varying types and subtlety.

**Why novel formal systems?**

Because novelty is the key to separating genuine rule-checking from pattern-matching. If the algebra doesn't exist in any training corpus, detecting a rule violation requires actually looking up the operation table and verifying the result. Pattern-matching from training data cannot help. 

**🤗 Why emojis?**

They tokenize reliably as single tokens across all major model families, they're visually distinct for human readability, and critically they carry no mathematical semantics. The model can't fall back on implicit knowledge of what "+" or "×" means — it must reason purely from the rules provided in-context.

## Dataset Construction

Each benchmark instance is assembled in four stages: **system generation**, **chain construction**, **error injection**, and **validation**. Every stage is deterministically seeded for full reproducibility.

### Stage 1 -- System Generation

A formal system consists of:

1. **Symbols** -- *n* emoji sampled from a curated pool of 30 (chosen for single-token encoding across all major model families and zero mathematical connotation).
2. **Base operations** -- *n* x *n* lookup tables defining binary operators (⊕, ⊗, ...). When the system requires transformation rules, tables are constructed over the cyclic group Z/nZ with a randomized label assignment. This guarantees automorphisms exist while making the table appear arbitrary to the model. When no transforms are needed, purely random magma tables are used for maximum novelty.
3. **Derived operations** -- defined compositionally in terms of base operations (e.g., `x ⊗ y = (x ⊕ x) ⊕ y`). Three templates are available: `compose_left`, `double_left`, and `inv_compose`.
4. **Transformation rules** -- symbol-level permutations that satisfy the distribution property over all base operations: `t(a ⊕ b) = t(a) ⊕ t(b)`. These are drawn from powers of a random permutation of sufficient order, ensuring they form a consistent cyclic automorphism group.

Difficulty levels control the system complexity:

| Level    | Symbols | Base Ops | Derived Ops | Transforms | Target Steps |
|----------|---------|----------|-------------|------------|--------------|
| Easy     | 3       | 1        | 0           | 0          | 4            |
| Medium   | 4       | 1        | 1           | 1          | 4            |
| Hard     | 5       | 2        | 1           | 1          | 4            |
| Expert   | 6       | 2        | 2           | 2          | 4            |

### Stage 2 -- Chain Construction

A derivation chain is a sequence of rewrite steps reducing a compound expression to a single symbol. The generator:

1. Builds a random expression tree targeting the desired step count.
2. Reduces it via **leftmost-innermost** evaluation order -- at each step, the deepest leftmost reducible subexpression is selected and rewritten.
3. Derived operations are expanded into their base-op definitions before reduction, producing an explicit "by definition of ⊗" step followed by base-op table lookups.

Each step is rendered as a **full-expression rewrite** (the entire expression before and after the reduction), keeping the derivation unambiguous even when identical subexpressions appear multiple times.

### Stage 3 -- Error Injection

Each base system produces up to four variants: one **clean** (no error) and three with a single injected error. Error types are summarized below.

| Type       | Name                    | What Changes                                                                                   | Where Injected     | Why It’s Hard                                                                                      |
|------------|-------------------------|-----------------------------------------------------------------------------------------------|--------------------|----------------------------------------------------------------------------------------------------|
| `E-RES`    | Wrong result            | The output symbol of a step is swapped to a different symbol. The cited rule and operands are correct. | Final step only    | The model must verify the table lookup rather than relying on surface plausibility.                 |
| `E-INV`    | Invented rule           | A step cites a plausible-looking rule that does not exist in the system (e.g., a nonexistent transform or operator). | Any step           | The model must check rule existence, not just whether the rewrite "looks right."                    |
| `E-CASC`   | Cascading wrong result  | One early step is made wrong, then all subsequent steps are **recomputed** to be locally valid given the wrong intermediate. | Any non-final step | Downstream steps are individually correct -- only the root violation is wrong. The model must trace the error back to its source. |

### Stage 4 -- Validation

Every generated system is validated before emission:

- All base operation tables are checked for completeness (every symbol pair has an entry) and closure (every result is a member of the symbol set).
- Every transformation rule is verified to satisfy the distribution property over all base operations it claims to distribute over.
- Every derived operation is evaluated exhaustively over all symbol pairs to confirm it produces valid results.

The released dataset (`emoji-bench-mixed-2000`) contains **1,998 instances** across all four difficulty levels, split roughly evenly: 500 clean, 500 E-RES, 500 E-INV, and 498 E-CASC.

## Evaluation Protocol

### Prompt Structure

Each prompt has three sections delivered as the user message:

```
Below is a formal system called "{system name}".

=== RULES ===
{symbol set, operation tables, derived op definitions, transform mappings}

=== YOUR WORKING OUT ===
Start: {expression}

Step 1: {before} = {after}    [by {rule}]
Step 2: ...
...

Result: {final symbol}

=== TASK ===
Check whether your working out contains an error.

Return:
1. `has_error`: yes or no
2. `error_step`: the first incorrect step number, or `null` if there is no error
```

The framing as "your working out" is deliberate -- it positions the model as auditing its own prior reasoning, which is the metacognitive setting we want to measure.

A system prompt instructs the model to return only the two structured fields without explanation:

> *You are reviewing your own prior working out in a formal system. Return only the structured fields requested by the schema: has_error (true/false), error_step (integer or null). Do not explain your answer.*

### Output Format

All models are called with **structured output** enforcement:

- **OpenAI** -- Responses API with Pydantic-based `parse`, automatic retry on `max_output_tokens` truncation (up to 4 retries with exponential token doubling).
- **Anthropic** -- Messages API with `json_schema` output config. Extended thinking disabled for all runs.
- **Mistral** -- Chat Completions API with `response_format: json_object` and `temperature: 0`.

### Scoring

Each prediction is scored on three binary metrics:

| Metric               | Definition                                                              |
|----------------------|-------------------------------------------------------------------------|
| `has_error_correct`  | Model correctly identifies whether the chain contains an error.          |
| `error_step_correct` | Model identifies the correct first error step (or `null` for clean).     |
| `joint_correct`      | Both of the above are correct. This is the primary metric.               |

Scores are aggregated overall and broken down by difficulty level and error type.

### Models

| Model              | Provider  | Reasoning       | Max Output Tokens |
|--------------------|-----------|-----------------|-------------------|
| GPT-4.1 mini       | OpenAI    | --              | 512               |
| GPT-5.4            | OpenAI    | medium effort   | 512               |
| GPT-5.4 mini       | OpenAI    | medium effort   | 512               |
| GPT-5.4 nano       | OpenAI    | medium effort   | 2048              |
| Claude Sonnet 4.6  | Anthropic | thinking off    | 512               |
| Claude Haiku 4.5   | Anthropic | thinking off    | 512               |
| Mistral Large 3    | Mistral   | --              | 512               |
| Mistral Medium 3.1 | Mistral   | --              | 512               |

All models use their provider's structured output mechanism. Reasoning-capable models (GPT-5.4 family) are set to medium effort; Anthropic extended thinking is disabled to keep the comparison uniform. GPT-5.4 nano's token limit is raised to 2048 because 512 proved insufficient for reliable structured output completion on this task.