from __future__ import annotations

import random
from dataclasses import replace

from emoji_bench.benchmark_types import ErrorInfo, ErrorType
from emoji_bench.chain_types import ChainStep, DerivationChain
from emoji_bench.expressions import SymbolLiteral
from emoji_bench.types import FormalSystem, Symbol


def get_wrong_result_eligible_steps(chain: DerivationChain) -> tuple[ChainStep, ...]:
    """Return steps eligible for non-cascading wrong-result injection.

    Because prompts are rendered as full-expression rewrites, changing any
    non-terminal step would break step-to-step continuity unless later steps
    were recomputed. For the minimal non-cascading E-RES case, only the
    terminal result-bearing step is eligible.
    """
    if not chain.steps:
        return ()

    last_step = chain.steps[-1]
    if last_step.result_symbol is None:
        return ()
    if not isinstance(last_step.after, SymbolLiteral):
        return ()
    return (last_step,)


def inject_wrong_result(
    chain: DerivationChain,
    system: FormalSystem,
    *,
    step_number: int | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> tuple[DerivationChain, ErrorInfo]:
    """Inject a single wrong-result error into a correct derivation chain."""
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    if seed is not None:
        rng = random.Random(seed)
    elif rng is None:
        rng = random.Random()

    eligible_steps = get_wrong_result_eligible_steps(chain)
    if not eligible_steps:
        raise ValueError("No eligible steps for wrong-result injection")

    if step_number is None:
        target = rng.choice(eligible_steps)
    else:
        matches = [step for step in eligible_steps if step.step_number == step_number]
        if not matches:
            raise ValueError(
                f"Step {step_number} is not eligible for wrong-result injection"
            )
        target = matches[0]

    assert target.result_symbol is not None
    assert isinstance(target.after, SymbolLiteral)

    wrong_choices = [sym for sym in system.symbols if sym != target.result_symbol]
    injected_result = rng.choice(wrong_choices)
    injected_after = SymbolLiteral(injected_result)

    mutated_step = replace(
        target,
        result_symbol=injected_result,
        after=injected_after,
    )
    mutated_steps = chain.steps[:-1] + (mutated_step,)
    mutated_chain = DerivationChain(
        starting_expression=chain.starting_expression,
        steps=mutated_steps,
        final_result=injected_result,
        seed=chain.seed,
    )

    error_info = ErrorInfo(
        error_type=ErrorType.E_RES,
        step_number=target.step_number,
        correct_result=target.result_symbol,
        injected_result=injected_result,
        correct_after=target.after,
        injected_after=injected_after,
        original_chain=chain,
    )
    return mutated_chain, error_info
