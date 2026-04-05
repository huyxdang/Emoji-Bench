---
pretty_name: Emoji-Bench
task_categories:
- text-generation
language:
- en
---

# huyxdang/emoji-bench-e-reconv-1000

This dataset contains prompt-only benchmark instances for Emoji-Bench.

## Schema

- `example_id`: unique row id
- `base_id`: shared id across clean/error variants of the same underlying problem
- `split`: train / validation / test
- `difficulty`: easy / medium / hard / expert
- `condition`: clean or error_injected
- `error_type`: null or an injected error label such as E-RES, E-INV, E-CASC, or E-RECONV
- `has_error`: whether the prompt contains an injected error
- `prompt`: full benchmark prompt
- `actual_step_count`: realized number of derivation steps
- `target_step_count`: requested target length used during generation
- `expected_error_step`: ground-truth step with the injected error, or null on clean rows
- `system_json`: JSON serialization of the underlying formal system
- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility

## Counts

- total_examples: 1000
- split_counts: {"train": 0, "validation": 0, "test": 1000}
- difficulty_counts: {"easy": 250, "medium": 250, "hard": 250, "expert": 250}
- condition_counts: {"error_injected": 1000}
- error_type_counts: {"E-RECONV": 1000}
- generator_commit: 20a04f2f1ccbaa9ab203177ea453e19ad8c26541

## Load

```python
from datasets import load_dataset

ds = load_dataset("huyxdang/emoji-bench-e-reconv-1000")
print(ds)
```
