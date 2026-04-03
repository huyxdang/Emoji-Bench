I wanna setup a website. Would make this thing much easier to show. Especially the Initial results. More clear too. Will host on Github pages.io. 

> https://github.com/huyxdang/Emoji-Bench
> 

> https://huggingface.co/datasets/huyxdang/emoji-bench-mixed-2000
> 

> [Measuring Progress Toward AGI - Cognitive Abilities](https://www.kaggle.com/competitions/kaggle-measuring-agi) (Meta-recognition track)
> 

## Problem

A central open question in LLM metacognition is whether models genuinely detect errors or merely pattern-match on what mistakes "look like" in training data. Existing benchmarks all operate in domains heavily represented in pretraining (arithmetic, algebra, code), making genuine constraint evaluation indistinguishable from memorized error-correction patterns.

We set out to measure genuine metacognitive reasoning in LLMs. 

---

## Solution

Emoji-Bench constructs **completely novel formal systems** — invented algebras built from emoji symbols (🦩 ⊕ 🧲 = 🪣) with procedurally generated operation tables, derived operations, and transformation rules. Models are given the rules and a multi-step derivation chain, then asked to verify whether each step correctly applies the rules. Some chains contain injected errors of varying types and subtlety.

**Why novel formal systems?**

Because novelty is the key to separating genuine rule-checking from pattern-matching. If the algebra doesn't exist in any training corpus, detecting a rule violation requires actually looking up the operation table and verifying the result. Pattern-matching from training data cannot help. 

**🤗 Why emojis?**

They tokenize reliably as single tokens across all major model families, they're visually distinct for human readability, and critically they carry no mathematical semantics. The model can't fall back on implicit knowledge of what "+" or "×" means — it must reason purely from the rules provided in-context.

---

## Dataset Construction

Each instance is built in four deterministically seeded stages. 

### Stage 1 — System Generation

A formal system consists of **symbols** (3–6 emoji from a curated pool of 30, chosen for single-token encoding and zero mathematical connotation), **base operations** (n × n lookup tables defining binary operators like ⊕), **derived operations** (defined compositionally, e.g., `x ⊗ y = (x ⊕ x) ⊕ y`), and **transformation rules** (unary permutations satisfying `t(a ⊕ b) = t(a) ⊕ t(b)`).

Difficulty scales by adding components while holding chain length constant at 4 steps:

| Level | Symbols | Base Ops | Derived Ops | Transforms |
| --- | --- | --- | --- | --- |
| Easy | 3 | 1 | 0 | 0 |
| Medium | 4 | 1 | 1 | 1 |
| Hard | 5 | 2 | 1 | 1 |
| Expert | 6 | 2 | 2 | 2 |
- More **Symbols** = Larger lookup tables to verify against
- More **Base Operations** = more rules to track
- **Derived Operations** add an inference layer (expand the definition, then check).
- **Transformations** require the most complex verification

### Stage 2 — Chain Construction

A derivation chain reduces a compound expression to a single symbol via **leftmost-innermost** evaluation — at each step, the deepest leftmost reducible subexpression is rewritten. Each step shows the full expression before and after reduction, keeping the derivation unambiguous.

### Stage 3 — Error Injection

Each system produces up to four variants: one **clean** and three with a single injected error.

| Type | What Changes | Why It's Hard |
| --- | --- | --- |
| `E-RES` | Output symbol swapped at the final step; cited rule and operands remain correct | Must verify the table lookup, not just surface plausibility |
| `E-INV` | A step cites a plausible rule that doesn't exist in the system | Must check rule existence, not just whether the rewrite looks right |
| `E-CASC` | One early step is wrong, then all subsequent steps are recomputed to be locally valid | Every downstream step checks out individually — must trace back to the root violation |

### Stage 4 — Validation

Every system is validated for operation table completeness and closure, transformation distribution property correctness, and derived operation consistency. The released dataset contains **1,998 instances** split evenly across conditions: 500 clean, 500 E-RES, 500 E-INV, 498 E-CASC.

### Example: Medium `E-CASC` Instance

```python
Symbols: {🪸, 🪩, 🧄, 🍄}

Operation ⊕ (defined by table):

| ⊕ | 🪸 | 🪩 | 🧄 | 🍄 |
|---|---|---|---|---|
| 🪸 | 🧄 | 🍄 | 🪩 | 🧄 |
| 🪩 | 🧄 | 🍄 | 🍄 | 🪸 |
| 🧄 | 🍄 | 🪩 | 🪸 | 🪸 |
| 🍄 | 🪸 | 🧄 | 🪩 | 🪩 |

Derived operation ⊗:
x ⊗ y = (x ⊕ x) ⊕ y

Transformation "inv":
  inv(🪸) = 🪩
  inv(🪩) = 🪸
  inv(🧄) = 🍄
  inv(🍄) = 🧄
  Distribution property: inv(x ⊕ y) = inv(x) ⊕ inv(y)

Start: (inv(🧄) ⊗ inv(🍄))

Step 1: (inv(🧄) ⊗ inv(🍄)) → (🍄 ⊗ inv(🍄))     [by inv]
Step 2: (🍄 ⊗ inv(🍄)) → (🍄 ⊗ 🍄)               [by inv]  ← ERROR
Step 3: (🍄 ⊗ 🍄) → ((🍄 ⊕ 🍄) ⊕ 🍄)             [by definition of ⊗]
Step 4: ((🍄 ⊕ 🍄) ⊕ 🍄) → (🪩 ⊕ 🍄)             [by ⊕ table]
Step 5: (🪩 ⊕ 🍄) → 🪸                           [by ⊕ table]

Result: 🪸
```

---

## Evaluation Protocol

### Prompt Structure

Each prompt has three sections delivered as the user message:

```
Below is a formal system called "{system name}".

=== RULES ===
{symbol set, operation tables, derived op definitions, transform mappings}

=== YOUR WORKING OUT ===
Start: {expression}

Step 1: {before} → {after}    [by {rule}]
Step 2: ...
...

Result: {final symbol}

=== TASK ===
Check whether your working out contains an error.

Return:
1. `has_error`: yes or no
2. `error_step`: the first incorrect step number, or `null` if there is no error
```

The framing as "your working out" is deliberate — it positions the model as auditing its own prior reasoning, which is the metacognitive setting we want to measure.

A system prompt instructs the model to return only the two structured fields without explanation:

> *You are reviewing your own prior working out in a formal system. Return only the structured fields requested by the schema: has_error (true/false), error_step (integer or null). Do not explain your answer.*
> 

### Output Format

All models are called with **structured output** enforcement:

- **OpenAI** — Responses API with Pydantic-based `parse`, automatic retry on `max_output_tokens` truncation (up to 4 retries with exponential token doubling).
- **Anthropic** — Messages API with `json_schema` output config. Extended thinking disabled for all runs.
- **Mistral** — Chat Completions API with `response_format: json_object` and `temperature: 0`.

### Scoring

Each prediction is scored on three binary metrics:

| Metric | Definition |
| --- | --- |
| `has_error_correct` | Model correctly identifies whether the chain contains an error. |
| `error_step_correct` | Model identifies the correct first error step (or `null` for clean). |
| `joint_correct` | Both of the above are correct. This is the primary metric. |

Scores are aggregated overall and broken down by difficulty level and error type.

### Models

| Model | Provider | Reasoning | Max Output Tokens |
| --- | --- | --- | --- |
| GPT-4.1 mini | OpenAI | — | 2048 |
| GPT-5.4 | OpenAI | medium effort | 2048 |
| GPT-5.4 mini | OpenAI | medium effort | 2048 |
| GPT-5.4 nano | OpenAI | medium effort | 2048 |
| Claude Sonnet 4.6 | Anthropic | thinking off | 2048 |
| Claude Haiku 4.5 | Anthropic | thinking off | 2048 |
| Mistral Large 3 | Mistral | — | 2048 |
| Mistral Medium 3.1 | Mistral | — | 2048 |



# Initial Results

Results on the full 1,998-example `emoji-bench-mixed-2000` dataset across 8 models (3 OpenAI reasoning, 1 OpenAI non-reasoning, 2 Anthropic, 2 Mistral).

## Overall

| Model | Accuracy | FP Rate |
|-------|----------|---------|
| **GPT-5.4** | **87.5%** | **0.0%** |
| GPT-5.4-nano | 74.3% | 2.6% |
| GPT-5.4-mini | 71.7% | 2.6% |
| GPT-4.1 mini | 49.5% | 55.2% |
| Mistral Large 3 | 48.7% | 40.8% |
| Mistral Medium 3.1 | 48.7% | 25.6% |
| Claude Haiku 4.5 | 48.4% | 61.4% |
| Claude Sonnet 4.6 | 48.4% | 15.0% |

## By Difficulty

| Model | Easy | Medium | Hard | Expert |
|-------|------|--------|------|--------|
| GPT-5.4 | 96.2% | 85.2% | 84.4% | 84.2% |
| GPT-5.4-nano | 74.5% | 75.2% | 74.0% | 73.4% |
| GPT-5.4-mini | 71.7% | 72.1% | 72.4% | 70.6% |
| GPT-4.1 mini | 51.5% | 50.1% | 50.2% | 46.2% |
| Mistral Large 3 | 49.7% | 51.3% | 47.0% | 47.0% |
| Mistral Medium 3.1 | 51.1% | 49.1% | 47.0% | 47.4% |
| Claude Haiku 4.5 | 44.9% | 52.3% | 49.6% | 46.8% |
| Claude Sonnet 4.6 | 49.3% | 50.9% | 44.2% | 49.4% |

## By Error Type

| Model | Clean | E-RES | E-INV | E-CASC |
|-------|-------|-------|-------|--------|
| GPT-5.4 | 100% | 100% | 50.0% | 100% |
| GPT-5.4-nano | 97.4% | 97.0% | 7.4% | 99.8% |
| GPT-5.4-mini | 97.4% | 99.2% | 2.6% | 99.0% |
| GPT-4.1 mini | 44.8% | 77.6% | 77.6% | 71.7% |
| Mistral Large 3 | 59.2% | 56.2% | 85.0% | 56.6% |
| Mistral Medium 3.1 | 74.4% | 58.0% | 71.4% | 47.8% |
| Claude Haiku 4.5 | 38.6% | 85.4% | 78.8% | 79.9% |
| Claude Sonnet 4.6 | 85.0% | 74.8% | 34.4% | 50.4% |

## Interpretations

**GPT-5.4 is far ahead of the field.** It achieves 87.5% accuracy with zero false positives. This is the only model that appears to perform genuine step-by-step constraint checking across all error types.

**E-INV (invented rule) is the critical discriminator.** The GPT-5.x reasoning family shows a dramatic failure mode on invented-rule errors: GPT-5.4-mini detects only 2.6% and GPT-5.4-nano only 7.4%. Even GPT-5.4 drops to 50%. These models excel at verifying that a computation was done correctly (E-RES, E-CASC), but struggle to verify that a cited rule *actually exists* in the formal system. This suggests reasoning models may be verifying forward computation rather than cross-referencing the rule set.

**Non-reasoning models are false-positive machines.** Claude Haiku (61.4% FP rate), GPT-4.1 mini (55.2%), and Mistral Large (40.8%) frequently hallucinate errors in clean derivation chains. Their accuracy collapses to ~48-50% because false alarms on clean rows offset correct detections on error rows.

**Claude Sonnet 4.6 is conservative but under-sensitive.** It has the lowest FP rate among non-GPT-5.x models (15%), but misses ~47% of actual errors. It detects only 34.4% of E-INV errors, making it even more blind to invented rules than the smaller reasoning models.

**E-CASC (cascading errors) is easy for reasoning models, harder for others.** GPT-5.4/5.4-mini/5.4-nano detect cascading errors at 99-100%. The chain recomputation that makes E-CASC locally consistent after the root error does not fool models that verify from the start. Non-reasoning models detect E-CASC at 48-80%, suggesting they rely on local consistency checks and miss the upstream violation.

**Difficulty scaling is modest but real.** GPT-5.4 drops ~12pp from easy (96.2%) to expert (84.2%). Other models show only 2-7pp spread -- they are near their floor regardless of problem complexity, consistent with the hypothesis that they are not performing systematic constraint evaluation.
