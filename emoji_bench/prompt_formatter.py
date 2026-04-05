from __future__ import annotations

from emoji_bench.chain_types import ChainStep, DerivationChain
from emoji_bench.expressions import expr_to_str_with_system
from emoji_bench.formatter import format_system_for_prompt
from emoji_bench.types import FormalSystem


def format_step(step: ChainStep, system: FormalSystem) -> str:
    """Format a single derivation step.

    Steps are rendered as full-expression rewrites to keep the derivation
    unambiguous even when the same local subexpression appears multiple times.
    """
    before_str = expr_to_str_with_system(step.before, system)
    after_str = expr_to_str_with_system(step.after, system)
    return f"Step {step.step_number}: {before_str} = {after_str}    [by {step.rule_used}]"


def format_chain(chain: DerivationChain, system: FormalSystem) -> str:
    """Format a derivation chain as numbered steps."""
    start_str = expr_to_str_with_system(chain.starting_expression, system)
    lines = [f"Start: {start_str}", ""]

    for step in chain.steps:
        lines.append(format_step(step, system))

    lines.append(f"\nResult: {chain.final_result.emoji}")
    return "\n".join(lines)


TASK_PROMPT = """\
Check whether your working out contains an error. It may or may not contain an error. 

Return:
1. `has_error`: yes or no
2. `error_step`: the first incorrect step number, or `null` if there is no error"""


def format_benchmark_prompt(system: FormalSystem, chain: DerivationChain) -> str:
    """Format the complete benchmark prompt: rules + derivation + task."""
    rules = format_system_for_prompt(system)
    derivation = format_chain(chain, system)

    return (
        f'Below is a formal system called "{system.name}".\n\n'
        f"=== RULES ===\n{rules}\n\n"
        f"=== YOUR WORKING OUT ===\n{derivation}\n\n"
        f"=== TASK ===\n{TASK_PROMPT}"
    )
