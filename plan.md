# Emoji-Bench

## A Benchmark for Testing Error Detection in LLMs

---

## 1. Motivation

When a language model encounters an error in its own reasoning (or in reasoning it is asked to evaluate), two fundamentally different internal processes could be responsible for its behavior:

**Circuit 1 — Error Monitor.** The model computes a mismatch signal: "this step is inconsistent with the stated constraints." It then acts on that signal to flag or correct the error. This constitutes genuine metacognition — the model is evaluating correctness, not just fluency.

**Circuit 2 — Coherence Engine.** The model only ever optimizes for "what is the most likely next token given the preceding sequence." When it appears to detect errors, it is pattern-matching on surface features of text that resembled mistakes in training data. When it fails to detect errors, it generates fluent justification that preserves the mistake — confabulation masquerading as reasoning.

From the outside, these two circuits are often indistinguishable. A model that says "wait, step 3 is wrong because..." might be genuinely checking constraints or might be producing a high-likelihood token sequence that follows certain textual patterns of error correction.

Emoji-Bench is designed to **pull these two circuits apart**.

### Why Novelty is the Key

Existing benchmarks for self-correction and error detection use domains the model has seen extensively in training: arithmetic, algebra, logic puzzles, code. High performance on these benchmarks cannot distinguish Circuit 1 from Circuit 2, because the model has been exposed to thousands of examples of errors and corrections in these domains during training.

Emoji-Bench eliminates this confound by constructing **completely novel formal systems** — invented algebras using emoji symbols, with grammars and transformation rules that do not exist in any training corpus. If a model can detect rule violations in a system it has never seen before, it must be performing genuine constraint evaluation against the rules provided in-context. Pattern-matching from training data cannot help.

### Conceptual Relationship to ARC-AGI

ARC-AGI (Chollet, 2019) tests whether models can generalize to novel abstract *patterns* they haven't encountered in training. Emoji-Bench tests whether models can enforce novel abstract *rules* they haven't encountered in training. Both are fundamentally about reasoning over unfamiliar structure rather than retrieval from memory.

ARC-AGI asks: "Can you solve novel problems?"
Emoji-Bench asks: "Can you notice when you're solving them wrong?"

Emoji-Bench can be understood as the **metacognitive counterpart** to ARC-AGI.

---

## 2. Benchmark Overview

### 2.1 High-Level Structure

1. **Generate** a novel formal system with arbitrary symbols, operations, and transformation rules.
2. **Construct** a multi-step reasoning chain (derivation) within that system.
3. **Optionally inject** one or more errors into the chain — steps that violate the system's rules but are stylistically and syntactically indistinguishable from correct steps.
4. **Present** the system rules and the chain to the model under test.
5. **Ask** the model to verify whether each step correctly applies the rules, and if not, identify and explain the violation.
6. **Score** the model's response against a rubric that evaluates detection, localization, and explanation quality.

### 2.2 What Emoji-Bench Measures

Emoji-Bench produces a multidimensional profile of a model's error-detection capability:

- **Detection rate** — Can the model tell that something is wrong?
- **Localization accuracy** — Can it identify *which* step is wrong?
- **Explanation validity** — Can it correctly articulate *why* it is wrong?
- **False positive rate** — Does it hallucinate errors in valid chains?
- **Position bias** — Does detection degrade depending on where the error appears?
- **Coherence resistance** — Does detection survive when post-error steps are fluent and internally consistent?
- **Circuit 1 vs. Circuit 2 separation** — Through control conditions, does the evidence point toward genuine constraint checking or surface pattern matching?

---

## 3. Formal System Design

### 3.1 System Components

Each novel formal system consists of:

| Component | Description | Example |
|---|---|---|
| **Symbol set** | 3–6 emoji symbols with no mathematical semantics | {🦩, 🧲, 🪣} |
| **Base operation(s)** | 1–2 binary operations defined by an explicit operation table | ⊕ defined by a 3×3 table |
| **Derived operation(s)** | 1–2 operations defined in terms of the base operation(s) | x ⊗ y = (x ⊕ y) ⊕ x |
| **Transformation rules** | 1–2 unary transformations with distribution properties | inv(x): 🦩→🧲, 🧲→🪣, 🪣→🦩; inv(x ⊕ y) = inv(x) ⊕ inv(y) |

### 3.2 Concrete Example: "Zelta Algebra"

**Symbols:** {🦩, 🧲, 🪣}

**Operation ⊕ (defined by table):**

| ⊕ | 🦩 | 🧲 | 🪣 |
|---|---|---|---|
| **🦩** | 🧲 | 🪣 | 🦩 |
| **🧲** | 🪣 | 🦩 | 🧲 |
| **🪣** | 🦩 | 🧲 | 🪣 |

**Derived operation:** x ⊗ y = (x ⊕ y) ⊕ x

**Transformation "inv":**
- inv(🦩) = 🧲
- inv(🧲) = 🪣
- inv(🪣) = 🦩
- Distribution property: inv(x ⊕ y) = inv(x) ⊕ inv(y)

### 3.3 Procedural System Generation

To ensure generality, Emoji-Bench requires a **system generator** that produces a unique formal system for each test instance (or batch of instances). The generator should:

1. **Sample a symbol set** of size $n \in \{3, 4, 5, 6\}$ from the curated emoji pool (see Section 3.5).
2. **Generate an operation table** — a random $n \times n$ table mapping pairs of symbols to symbols. Optionally enforce algebraic properties (closure is automatic; associativity, commutativity, or existence of identity/inverses can be toggled to control difficulty).
3. **Define 0–2 derived operations** using templates such as:
   - $x \circledast y = (x \oplus y) \oplus x$
   - $x \circledast y = \text{inv}(x \oplus y)$
   - $x \circledast y = (x \oplus x) \oplus y$
4. **Define 0–2 transformation rules** with explicit mappings and distribution properties.
5. **Validate internal consistency** — verify that all derived operations and transformation rules produce deterministic results given the base operation table. Discard and regenerate if inconsistencies are found.

### 3.4 Design Constraints

- **No mathematical semantics.** Symbols must not carry implicit mathematical or logical associations. Emoji are chosen specifically to be semantically distant from formal reasoning (see Section 3.5).
- **Determinism.** Every expression in the system must have exactly one correct result. Ambiguity in the rules would make the benchmark unmeasurable.
- **Complexity control.** The number of symbols, operations, and rules should be parameterized so that benchmark difficulty can be scaled systematically.

### 3.5 Emoji Symbol Pool

Emoji-Bench uses emoji as symbols because they offer several advantages over abstract glyphs (▲, ●, ■):

**Tokenization reliability.** Most common emoji are represented as single tokens in modern tokenizers (GPT, Claude, Gemma, Llama). Abstract Unicode symbols like ▲ or ■ can tokenize unpredictably — sometimes as a single token, sometimes split into multi-byte sequences — introducing unwanted variance across models.

**Visual distinctiveness.** Emoji are immediately visually distinguishable, making the benchmark easier for humans to read, debug, validate, and present in papers or demos.

**Semantic distance from mathematics.** By curating a pool of emoji that have no mathematical or logical associations, we ensure the model cannot leverage implicit semantic priors.

**Curated Pool (~30 emoji, randomized per system):**

| Category | Emoji |
|---|---|
| Animals | 🦩 🐙 🦔 🪼 🦎 🐌 |
| Objects | 🧲 🪣 🪆 🧿 🪤 🪩 |
| Nature | 🍄 🫧 🪸 🪻 🌵 🪨 |
| Food | 🧁 🫐 🥟 🪺 🧄 🫑 |
| Misc | 🪬 🧊 🪈 🪭 🧶 🪵 |

**Selection criteria:**
- Must tokenize as a single token across major model families (GPT-4, Claude, Llama 3, Gemma 2). Verify empirically before finalizing the pool.
- Must have no mathematical, logical, or ordering connotation. Excluded: ➕, ✖️, ⭐, 🔢, 🏆, 🥇, 🥈, 🥉, or any emoji commonly used in puzzle/quiz contexts.
- Must be visually distinct from each other at small font sizes.
- Assignments are randomized per formal system — the same emoji can play different roles in different systems, preventing the model from learning fixed symbol-role mappings across benchmark instances.

**Contamination risk mitigation.** While emoji appear in training data (including emoji math puzzles on social media), the combination of (a) randomized symbol-role assignments, (b) procedurally generated operation tables, and (c) novel derived operations and transformations ensures that no specific system instance exists in any training corpus. The emoji are familiar tokens; the algebraic structures built from them are not.

---

## 4. Reasoning Chain Construction

### 4.1 Chain Structure

A reasoning chain is a sequence of $N$ steps (typically $N \in \{5, 7, 10, 15\}$) that derives a final result from a starting expression. Each step has the format:

```
Step K: [expression] = [result]    [by rule/operation name]
```

For example:

```
Start: (🦩 ⊕ 🧲) ⊗ 🪣

Step 1: 🦩 ⊕ 🧲 = 🪣                              [by ⊕ table]
Step 2: 🪣 ⊗ 🪣 = (🪣 ⊕ 🪣) ⊕ 🪣                    [by definition of ⊗]
Step 3: 🪣 ⊕ 🪣 = 🪣                               [by ⊕ table]
Step 4: 🪣 ⊕ 🪣 = 🪣                               [by ⊕ table]
Step 5: Result: 🪣
```

### 4.2 Chain Generation Process

1. Sample a starting expression of depth 2–4 (e.g., `(x ⊕ y) ⊗ inv(z)`).
2. Compute the correct derivation step by step using the formal system rules.
3. Record each intermediate step with its justification.
4. Verify the full chain programmatically — the generator must be a correct interpreter for the formal system.

### 4.3 Error Injection

For error-containing chains, select one step $K$ and modify it. The modification must:

- **Preserve syntactic form.** The step should look exactly like a valid step — same format, same rule citation style, a result that is a valid symbol in the system.
- **Violate a rule.** The stated result must be incorrect given the actual rules.
- **(🔴 Hard. Is this necessary?) Maintain post-error coherence.** All steps after step $K$ should be computed correctly *given the wrong result at step $K$*. This is critical — it means the chain reads fluently after the error. Only step $K$ itself is wrong; everything downstream is locally valid but globally wrong.

This design choice directly tests Circuit 1 vs. Circuit 2. Circuit 2 would see a coherent chain and accept it. Circuit 1 would flag the original violation at step $K$.

---

## 5. Error Taxonomy

Not all errors are equivalent. Emoji-Bench categorizes injected errors into types of increasing subtlety:

### 5.1 Error Types

| Type | Code | Description | Example |
|---|---|---|---|
| **Wrong operands** | `E-OP` | Correct rule, applied to the wrong inputs | Applies "🦩 ⊕ 🧲 = 🪣" but the expression has "🧲 ⊕ 🦩" |
| **Wrong result** | `E-RES` | Correct operands, correct rule cited, but the stated output is incorrect | "🦩 ⊕ 🧲 = 🧲" instead of "🦩 ⊕ 🧲 = 🪣" |
| **Invented rule** | `E-INV` | Applies a transformation or operation that was never defined in the system | Uses "double(x)" when no such operation exists |
| **Cascading error** | `E-CASC` | Error at step $K$; all subsequent steps are valid given the wrong result but wrong given the correct result | Step 3 is wrong, steps 4–8 are locally correct but globally wrong |
| **Subtle off-by-one** | `E-SUB` | Swaps to an adjacent entry in the operation table — result is "close" to correct | 🦩 ⊕ 🧲 = 🦩 instead of 🪣, where 🦩 is the result of a neighboring cell |

**Implementation status:** `E-RES`, `E-INV`, and `E-CASC` are implemented in the current codebase. `E-OP` and `E-SUB` are deferred for now. Under the current prompt format, derivation steps are shown as full-expression rewrites rather than explicit local reductions, which makes these two categories weakly separable from `E-RES` in model-visible terms. They should be revisited after adding a prompt mode that exposes the local subexpression being reduced.

### 5.2 Difficulty Ordering

From easiest to hardest (predicted):

1. `E-INV` — Invented rules are the most obvious because the model can check whether the cited operation exists.
2. `E-RES` — Requires looking up the operation table and checking the output.
3. `E-OP` — Requires verifying that the operands in the step match the expression being simplified. (Deferred)
4. `E-SUB` — Hardest to detect because the wrong result is "almost right." (Deferred)
5. `E-CASC` — Tests whether the model catches the root error or accepts the locally-coherent cascade.

This ordering is itself a hypothesis that Emoji-Bench can test empirically.

---

## 6. Evaluation Protocol

### 6.1 Task Prompt Template

```
Below is a formal system called "[SYSTEM_NAME]".

=== RULES ===
[FULL SPECIFICATION OF SYMBOLS, OPERATIONS, AND TRANSFORMATION RULES]

=== DERIVATION ===
[MULTI-STEP REASONING CHAIN]

=== TASK ===
Verify whether each step in the derivation correctly applies the rules
of the formal system. If any step contains an error, identify:
1. Which step contains the error
2. What rule was violated
3. What the correct result should be

If all steps are correct, state that the derivation is valid.
```

### 6.2 Experimental Conditions

Emoji-Bench requires four conditions to separate Circuit 1 from Circuit 2:

**Implementation status:** the current codebase supports Condition 1 (error-injected chains) and Condition 2 (clean chains). Conditions 3 and 4, along with the control conditions below, remain roadmap items.

**Condition 1: Core Evaluation (Error-Injected Chains)**
Chains with exactly one injected error. This is the primary test. Measures detection rate, localization accuracy, and explanation quality across all error types.

**Condition 2: Clean Chains (No Errors)**
Chains with no errors. Measures false positive rate. Essential for ensuring the model isn't just saying "step X is wrong" for every input.

**Condition 3: Suspicious-but-Correct Chains**
Chains where all steps are correct, but some steps have results that "look" like they could be wrong — e.g., a result that is the same as one of the operands, or a step that produces the identity element. If the model flags these, it's using surface heuristics (something "looks off") rather than checking rules.

**Condition 4: Familiar-Domain Mirror**
The same chain structures but using standard arithmetic instead of novel formal systems. Comparing performance on Condition 1 (novel systems) vs. Condition 4 (arithmetic) measures the contribution of training-data pattern matching. A large gap means the model relies heavily on memorized error patterns (Circuit 2). A small gap means the model may be doing genuine constraint evaluation (Circuit 1).

### 6.3 Control Conditions for Circuit Separation

**Control A: Rule Visibility**
Present the derivation chain *without* the rules — just the chain and the question "is this valid?" If the model still "detects" errors at above-chance rates without access to the rules, it is using surface heuristics, not constraint checking. Detection rate should drop to near chance when rules are removed.

**Control B: Fluency-Matched Distractors**
For each error, create two versions:
- **Syntactically awkward, semantically correct:** A step that is valid but uses unusual formatting or phrasing.
- **Syntactically fluent, semantically wrong:** A step that is invalid but reads perfectly.

If the model catches the awkward-but-correct step more often than the fluent-but-wrong step, it is keying on disfluency rather than logical violation.

**Control C: Coherence Pressure Gradient**
Vary the number of post-error steps: 0, 2, 5, 10 steps after the error, all locally valid given the wrong result. If detection rate drops as post-error chain length increases, the model's coherence drive is overriding its error signal — evidence for Circuit 2 dominance.

### 6.4 Scoring Rubric

Each model response is scored on three dimensions:

**Detection (binary):**
- 1 if the model states that the chain contains an error
- 0 if the model states the chain is valid (or fails to identify any error)

**Localization (categorical):**
- 2 if the model identifies the exact correct step
- 1 if the model identifies a step within ±1 of the correct step
- 0 otherwise

**Explanation Quality (rubric):**
- 3 — Correctly identifies the violated rule AND states the correct result AND explains why the stated result is wrong
- 2 — Correctly identifies the violated rule AND states the correct result, but explanation is incomplete
- 1 — Identifies the correct step but gives a vague or partially incorrect explanation
- 0 — Misidentifies the violation, gives a wrong explanation, or hallucinates an error on a clean chain

**Composite Score:**
$$\text{Emoji-Bench Score} = \frac{1}{N} \sum_{i=1}^{N} \left( \mathbb{1}[\text{detected}_i] \times \frac{\text{localization}_i}{2} \times \frac{\text{explanation}_i}{3} \right)$$

This multiplicative structure means the model gets zero credit if it fails at any stage — detecting an error but localizing it wrong, or localizing it correctly but explaining it incorrectly, both receive heavily penalized scores.

---

## 7. Scaling Dimensions

Emoji-Bench is designed to be parameterized along several axes to produce a difficulty gradient:

### 7.1 System Complexity

| Level | Symbols | Base Operations | Derived Operations | Transformations |
|---|---|---|---|---|
| **Easy** | 3 | 1 | 0 | 0 |
| **Medium** | 4 | 1 | 1 | 1 |
| **Hard** | 5 | 2 | 1 | 1 |
| **Expert** | 6 | 2 | 2 | 2 |

### 7.2 Chain Length

| Level | Steps |
|---|---|
| **Short** | 3–5 |
| **Medium** | 6–10 |
| **Long** | 11–15 |

### 7.3 Error Subtlety

| Level | Error Types Included |
|---|---|
| **Obvious** | E-INV |
| **Moderate** | E-RES, E-OP |
| **Subtle** | E-SUB, E-CASC |

### 7.4 Interaction Analysis

The most informative results come from interactions between dimensions:

- **Detection degrades with chain length but not system complexity** → Coherence pressure (Circuit 2 signal). The model's error detection is overwhelmed by the volume of coherent text, not by the difficulty of the rules.
- **Detection degrades with system complexity but not chain length** → Working memory / in-context learning limitation. The model struggles to hold complex rule systems in context, but its metacognitive process itself may be functional.
- **Detection degrades with error subtlety uniformly** → The model has a genuine error-detection mechanism but with limited precision.
- **Detection degrades only for E-CASC but not other types** → Strong evidence for Circuit 2. The model accepts post-hoc coherence as evidence of correctness.

---

## 8. Dataset Specification

### 8.1 Recommended Dataset Size

| Component | Count | Purpose |
|---|---|---|
| Unique formal systems | 50–100 | Ensure generalization across systems |
| Chains per system | 10–20 | Statistical power within each system |
| Error-injected chains | ~60% of total | Core measurement |
| Clean chains | ~25% of total | False positive calibration |
| Suspicious-but-correct chains | ~15% of total | Heuristic detection |
| **Total instances** | **500–2000** | Full benchmark |

### 8.2 Balance Requirements

- Error types should be roughly equally represented within the error-injected subset.
- Error positions should be uniformly distributed across chain positions (early, middle, late).
- System complexity levels should be balanced.
- Chain lengths should be balanced within each complexity level.

### 8.3 Splits

- **Validation set (10%):** For developing scoring pipelines and debugging.
- **Test set (90%):** For final evaluation. No model should be tuned on this set.

Because the systems are procedurally generated, there is no contamination risk — but the generation code and seed should be versioned so results are reproducible.

---

## 9. Expected Results and Interpretive Framework

### 9.1 Ideal Circuit 1 Profile

A model with genuine error monitoring would show:

- High detection rate across all error types (>90%)
- High localization accuracy (consistently identifies the exact step)
- High explanation quality (correctly cites the violated rule and the correct result)
- Low false positive rate (<5%)
- No position bias (detection rate flat across early/middle/late positions)
- No coherence pressure effect (detection rate stable regardless of post-error chain length)
- Small gap between novel-system and familiar-domain performance
- Near-zero detection when rules are removed (Control A)
- No fluency bias (Control B)

### 9.2 Ideal Circuit 2 Profile

A model relying purely on coherence / pattern-matching would show:

- Moderate-to-low detection rate, especially for subtle error types
- Reasonable localization when errors are detected (pattern-matching can still localize)
- Explanation quality that references vague "something seems off" rather than specific rule violations
- Non-trivial false positive rate (flags "suspicious-looking" correct steps)
- Strong position bias (errors late in chain missed more often)
- Strong coherence pressure effect (detection drops sharply with more post-error steps)
- Large gap between novel-system and familiar-domain performance
- Above-chance detection even when rules are removed (using surface heuristics)
- Fluency bias present (flags awkward-but-correct steps)

### 9.3 Reality: The Spectrum

Most models will fall somewhere between these profiles, potentially showing Circuit 1-like behavior for some error types (e.g., E-INV, E-RES) and Circuit 2-like behavior for others (e.g., E-CASC, E-SUB). This granularity is by design — Emoji-Bench is not trying to produce a single "metacognition score" but a detailed diagnostic profile.

---

## 10. Implementation Notes

### 10.1 System Generator Requirements

The system generator should be implemented as a standalone Python module that:

- Takes parameters: `n_symbols`, `n_base_ops`, `n_derived_ops`, `n_transformations`, `random_seed`
- Outputs: a fully specified formal system (symbol set, operation tables, derived operation definitions, transformation rules)
- Includes a built-in **interpreter** that can evaluate any expression in the system to its correct result
- Includes a **chain generator** that produces valid derivation chains of specified length
- Includes an **error injector** that modifies a specified step according to a specified error type
- Includes a **consistency validator** that verifies the system has no internal contradictions

### 10.2 Evaluation Pipeline

1. Generate formal systems and chains (with and without errors).
2. Format each instance as a prompt using the task template.
3. Run the model under test on all instances.
4. Parse model responses to extract: detection decision, localized step, and explanation.
5. Score each response using the rubric.
6. Aggregate scores across dimensions (error type, chain length, system complexity, position, etc.).
7. Run control conditions and compare.
8. Produce a diagnostic report.

### 10.3 Model Response Parsing

Model responses are free-form text. A reliable parser should:

- Detect whether the model claims the chain is valid or contains errors.
- Extract the step number(s) the model identifies as erroneous.
- Extract the model's explanation of the violation.
- Handle edge cases: models that hedge ("step 3 might be wrong"), models that identify multiple errors when only one exists, models that recompute the entire chain.

An LLM-based judge (using a separate, stronger model) may be necessary for scoring explanation quality, as rubric application to free-form text is non-trivial.

### 10.4 Statistical Analysis

- Use **McNemar's test** for pairwise comparison of detection rates between conditions (e.g., novel vs. familiar domain).
- Use **logistic regression** to model detection probability as a function of error type, position, chain length, and system complexity.
- Report **95% confidence intervals** on all aggregate metrics.
- For Circuit 1 vs. Circuit 2 separation, compute effect sizes (Cohen's d or odds ratios) for each control condition.

---

## 11. Extensions and Future Work

### 11.1 Internal Representation Probing

For open-weight models, Emoji-Bench can be extended with mechanistic interpretability:

- Train a linear probe on the model's residual stream activations at each step to predict "this step contains an error."
- Key question: does the probe detect errors the model *fails to self-correct in its output*?
- If yes: the model has the information internally (proto-Circuit 1) but doesn't act on it — suggesting Circuit 2's coherence pressure overrides an existing error signal.
- If no: the model genuinely does not represent the error, and all detection is surface-level.

### 11.2 Cross-Domain Transfer

Test whether error-detection ability in one novel formal system transfers to a structurally isomorphic error in a completely different novel system. Circuit 1 (if it exists as a general mechanism) should transfer. Circuit 2 should not, because the surface patterns differ.

### 11.3 Self-Generated Chains

Instead of presenting pre-built chains for verification, ask the model to *generate* a derivation and then verify its own work. This tests metacognition in the most naturalistic setting — can the model catch its own errors? This is harder to control experimentally but closer to the real-world use case.

### 11.4 Multi-Agent Error Propagation

Present a chain where "Agent A" generated steps 1–5 and "Agent B" must continue from step 5. Inject an error in Agent A's work. Does Agent B catch it before continuing, or does it build on the error? This extends Emoji-Bench into the multi-agent communication setting.

---

## 12. Summary

Emoji-Bench is a benchmark designed to answer a specific, fundamental question about large language models:

**When a model detects an error, is it performing genuine constraint evaluation (Circuit 1) — or is it pattern-matching on surface features of text that resembled mistakes in its training data (Circuit 2)?**

It achieves this by:

1. Eliminating training-data confounds through procedurally generated novel formal systems.
2. Requiring explanation-level evidence, not just binary detection.
3. Including control conditions that specifically separate Circuit 1 from Circuit 2.
4. Parameterizing difficulty across multiple dimensions to produce a detailed diagnostic profile.
5. Providing a rigorous scoring rubric and statistical analysis framework.

The benchmark produces not a single score but a **metacognitive profile** — a detailed map of where, how, and under what conditions a model's error-detection capabilities succeed or fail.

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **Circuit 1 (Error Monitor)** | A hypothesized internal mechanism that detects constraint violations through genuine evaluation |
| **Circuit 2 (Coherence Engine)** | The default autoregressive behavior of optimizing for fluent, high-likelihood continuations |
| **Formal system** | A set of symbols, operations, and rules that define a self-contained mathematical structure |
| **Derivation chain** | A sequence of steps that applies rules to transform a starting expression into a result |
| **Error injection** | The process of modifying a correct step to introduce a rule violation |
| **Cascading error** | An error whose consequences propagate through subsequent steps that are locally valid but globally wrong |
| **Coherence pressure** | The tendency of autoregressive models to maintain narrative consistency at the expense of correctness |
| **Position bias** | Variation in detection rate depending on where in the chain the error appears |

## Appendix B: Related Work

- **ARC-AGI** (Chollet, 2019) — Abstraction and Reasoning Corpus for general intelligence evaluation
- **BIG-Bench** (Srivastava et al., 2023) — Broad benchmark including some logical reasoning tasks
- **SelfCheckGPT** (Manakul et al., 2023) — Hallucination detection through self-consistency
- **LLM Self-Correction Literature** (Huang et al., 2024) — Survey of intrinsic self-correction capabilities
- **Calibration and Confidence Estimation** — AUROC-based evaluation of model confidence (related to IDK-style benchmarks)
