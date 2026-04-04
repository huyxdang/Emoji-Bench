---
pretty_name: Emoji-Bench
task_categories:
- text-generation
language:
- en
---

# emoji-bench-mixed-2000-numbers

This dataset contains prompt-only benchmark instances for Emoji-Bench.

## Schema

- `example_id`: unique row id
- `base_id`: shared id across clean/error variants of the same underlying problem
- `split`: train / validation / test
- `difficulty`: easy / medium / hard / expert
- `condition`: clean or error_injected
- `error_type`: null, E-RES, E-INV, or E-CASC
- `has_error`: whether the prompt contains an injected error
- `prompt`: full benchmark prompt
- `actual_step_count`: realized number of derivation steps
- `target_step_count`: requested target length used during generation
- `expected_error_step`: ground-truth step with the injected error, or null on clean rows
- `system_json`: JSON serialization of the underlying formal system
- `system_seed` / `chain_seed` / `error_seed`: generation metadata for reproducibility

## Counts

- total_examples: 1998
- split_counts: {"train": 0, "validation": 0, "test": 1998}
- difficulty_counts: {"easy": 499, "medium": 499, "hard": 500, "expert": 500}
- condition_counts: {"clean": 500, "error_injected": 1498}
- error_type_counts: {"clean": 500, "E-RES": 500, "E-INV": 500, "E-CASC": 498}
- generator_commit: 16cf7a6c6d80ff25496796cfc033a3715cd0fc86

## Load

```python
from datasets import load_dataset

ds = load_dataset("emoji-bench-mixed-2000-numbers")
print(ds)
```
