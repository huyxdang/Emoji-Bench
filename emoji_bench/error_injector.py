from __future__ import annotations

import random
from dataclasses import replace

from emoji_bench.benchmark_types import ErrorInfo, ErrorType
from emoji_bench.chain_generator import reduce_expression
from emoji_bench.chain_types import ChainStep, DerivationChain
from emoji_bench.expressions import SymbolLiteral
from emoji_bench.types import FormalSystem, Symbol


def _resolve_rng(
    *,
    rng: random.Random | None,
    seed: int | None,
) -> random.Random:
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    if seed is not None:
        return random.Random(seed)
    if rng is None:
        return random.Random()
    return rng


def _pick_wrong_symbol(correct: Symbol, symbols: tuple[Symbol, ...], rng: random.Random) -> Symbol:
    wrong_choices = [sym for sym in symbols if sym != correct]
    return rng.choice(wrong_choices)


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
    rng = _resolve_rng(rng=rng, seed=seed)

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

    injected_result = _pick_wrong_symbol(target.result_symbol, system.symbols, rng)
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


def get_cascading_eligible_steps(chain: DerivationChain) -> tuple[ChainStep, ...]:
    """Return steps eligible for cascading wrong-result injection.

    A cascading error must occur before the terminal step so that downstream
    reductions can be recomputed from the wrong intermediate expression.
    """
    if len(chain.steps) < 2:
        return ()
    return tuple(
        step
        for step in chain.steps[:-1]
        if step.result_symbol is not None
    )


def inject_cascading_wrong_result(
    chain: DerivationChain,
    system: FormalSystem,
    *,
    step_number: int | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> tuple[DerivationChain, ErrorInfo]:
    """Inject a wrong result and recompute all downstream steps from that point."""
    rng = _resolve_rng(rng=rng, seed=seed)

    eligible_steps = get_cascading_eligible_steps(chain)
    if not eligible_steps:
        raise ValueError("No eligible steps for cascading wrong-result injection")

    if step_number is None:
        target = rng.choice(eligible_steps)
    else:
        matches = [step for step in eligible_steps if step.step_number == step_number]
        if not matches:
            raise ValueError(
                f"Step {step_number} is not eligible for cascading wrong-result injection"
            )
        target = matches[0]

    assert target.result_symbol is not None

    injected_result = _pick_wrong_symbol(target.result_symbol, system.symbols, rng)
    injected_after = replace(
        target.after,
        symbol=injected_result,
    ) if isinstance(target.after, SymbolLiteral) else None
    if injected_after is None:
        injected_after = SymbolLiteral(injected_result)

    mutated_root = replace(
        target,
        result_symbol=injected_result,
        after=injected_after,
    )

    suffix = reduce_expression(injected_after, system)
    renumbered_suffix = tuple(
        replace(step, step_number=mutated_root.step_number + idx)
        for idx, step in enumerate(suffix, start=1)
    )

    prefix = chain.steps[: target.step_number - 1]
    mutated_steps = prefix + (mutated_root,) + renumbered_suffix

    if renumbered_suffix:
        final_step = renumbered_suffix[-1]
        assert isinstance(final_step.after, SymbolLiteral)
        final_result = final_step.after.symbol
    else:
        assert isinstance(mutated_root.after, SymbolLiteral)
        final_result = mutated_root.after.symbol

    mutated_chain = DerivationChain(
        starting_expression=chain.starting_expression,
        steps=mutated_steps,
        final_result=final_result,
        seed=chain.seed,
    )

    error_info = ErrorInfo(
        error_type=ErrorType.E_CASC,
        step_number=target.step_number,
        correct_result=target.result_symbol,
        injected_result=injected_result,
        correct_after=target.after,
        injected_after=injected_after,
        original_chain=chain,
    )
    return mutated_chain, error_info
