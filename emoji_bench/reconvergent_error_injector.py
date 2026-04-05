from __future__ import annotations

import random
from dataclasses import replace

from emoji_bench.benchmark_types import ErrorInfo, ErrorType
from emoji_bench.chain_generator import (
    find_leftmost_innermost,
    reduce_expression,
    replace_at_path,
)
from emoji_bench.chain_types import ChainStep, DerivationChain
from emoji_bench.expressions import Expression, SymbolLiteral
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


def _build_injected_after(step: ChainStep, injected_result: Symbol) -> Expression:
    path = find_leftmost_innermost(step.before)
    if path is None:
        raise ValueError("Cannot inject an error into a non-reducible step")
    return replace_at_path(step.before, path, SymbolLiteral(injected_result))


def _reduce_from(
    expr: Expression,
    system: FormalSystem,
) -> tuple[tuple[ChainStep, ...], Symbol]:
    suffix = reduce_expression(expr, system)
    if suffix:
        final_step = suffix[-1]
        assert isinstance(final_step.after, SymbolLiteral)
        return suffix, final_step.after.symbol

    assert isinstance(expr, SymbolLiteral)
    return suffix, expr.symbol


def get_reconvergent_eligible_steps(chain: DerivationChain) -> tuple[ChainStep, ...]:
    """Return non-terminal result-bearing steps eligible for reconvergence search."""
    if len(chain.steps) < 2:
        return ()
    return tuple(
        step
        for step in chain.steps[:-1]
        if step.result_symbol is not None
    )


def inject_reconvergent_wrong_result(
    chain: DerivationChain,
    system: FormalSystem,
    *,
    step_number: int | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> tuple[DerivationChain, ErrorInfo]:
    """Inject a wrong intermediate result whose recomputed suffix reconverges."""
    rng = _resolve_rng(rng=rng, seed=seed)

    eligible_steps = get_reconvergent_eligible_steps(chain)
    if not eligible_steps:
        raise ValueError("No eligible steps for reconvergent wrong-result injection")

    if step_number is None:
        targets = sorted(eligible_steps, key=lambda step: step.step_number)
    else:
        matches = [step for step in eligible_steps if step.step_number == step_number]
        if not matches:
            raise ValueError(
                f"Step {step_number} is not eligible for reconvergent wrong-result injection"
            )
        targets = matches

    for target in targets:
        assert target.result_symbol is not None
        wrong_choices = [sym for sym in system.symbols if sym != target.result_symbol]
        rng.shuffle(wrong_choices)

        for injected_result in wrong_choices:
            injected_after = _build_injected_after(target, injected_result)
            suffix, final_result = _reduce_from(injected_after, system)
            if final_result != chain.final_result:
                continue

            mutated_root = replace(
                target,
                result_symbol=injected_result,
                after=injected_after,
            )
            renumbered_suffix = tuple(
                replace(step, step_number=mutated_root.step_number + idx)
                for idx, step in enumerate(suffix, start=1)
            )
            prefix = chain.steps[: target.step_number - 1]
            mutated_steps = prefix + (mutated_root,) + renumbered_suffix
            mutated_chain = DerivationChain(
                starting_expression=chain.starting_expression,
                steps=mutated_steps,
                final_result=final_result,
                seed=chain.seed,
            )
            error_info = ErrorInfo(
                error_type=ErrorType.E_RECONV,
                step_number=target.step_number,
                correct_result=target.result_symbol,
                injected_result=injected_result,
                correct_after=target.after,
                injected_after=injected_after,
                original_chain=chain,
            )
            return mutated_chain, error_info

    raise ValueError(
        "Could not inject a reconvergent wrong result that preserves the final result"
    )
