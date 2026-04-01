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
from emoji_bench.generator import OPERATOR_SYMBOLS, TRANSFORM_NAMES
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


def _build_injected_after(step: ChainStep, injected_result: Symbol) -> Expression:
    """Rebuild step.after with only the reduced subexpression changed."""
    path = find_leftmost_innermost(step.before)
    if path is None:
        raise ValueError("Cannot inject an error into a non-reducible step")
    return replace_at_path(step.before, path, SymbolLiteral(injected_result))


def _reduce_from(
    expr: Expression,
    system: FormalSystem,
) -> tuple[tuple[ChainStep, ...], Symbol]:
    """Reduce an expression and return both the suffix steps and final symbol."""
    suffix = reduce_expression(expr, system)
    if suffix:
        final_step = suffix[-1]
        assert isinstance(final_step.after, SymbolLiteral)
        return suffix, final_step.after.symbol

    assert isinstance(expr, SymbolLiteral)
    return suffix, expr.symbol


def _available_rule_choices(system: FormalSystem) -> tuple[tuple[str, str], ...]:
    """Return all rule labels available in a system."""
    choices: list[tuple[str, str]] = []
    for op in system.base_operations:
        choices.append((f"{op.symbol_id} table", "base_op"))
    for dop in system.derived_operations:
        choices.append((f"definition of {dop.symbol_id}", "derived_op"))
    for transform in system.transformations:
        choices.append((transform.name, "transform"))
    return tuple(choices)


def _invented_rule_choices(system: FormalSystem) -> tuple[tuple[str, str], ...]:
    """Return plausible rule labels that are not defined in the system."""
    available_rules = {rule_used for rule_used, _ in _available_rule_choices(system)}
    used_op_symbols = {
        op.symbol_id for op in (*system.base_operations, *system.derived_operations)
    }
    used_transform_names = {transform.name for transform in system.transformations}

    invented_choices: list[tuple[str, str]] = []

    for op_symbol in OPERATOR_SYMBOLS:
        if op_symbol in used_op_symbols:
            continue
        invented_choices.append((f"{op_symbol} table", "base_op"))
        invented_choices.append((f"definition of {op_symbol}", "derived_op"))

    for transform_name in TRANSFORM_NAMES:
        if transform_name in used_transform_names:
            continue
        invented_choices.append((transform_name, "transform"))

    return tuple(
        (rule_used, rule_type)
        for rule_used, rule_type in invented_choices
        if rule_used not in available_rules
    )


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


def get_invented_rule_eligible_steps(
    chain: DerivationChain,
    system: FormalSystem,
) -> tuple[ChainStep, ...]:
    """Return steps where a nonexistent but plausible rule label can be cited."""
    if not _invented_rule_choices(system):
        return ()
    return chain.steps


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


def inject_invented_rule(
    chain: DerivationChain,
    system: FormalSystem,
    *,
    step_number: int | None = None,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> tuple[DerivationChain, ErrorInfo]:
    """Inject a citation to a rule that is not defined in the system."""
    rng = _resolve_rng(rng=rng, seed=seed)

    eligible_steps = get_invented_rule_eligible_steps(chain, system)
    if not eligible_steps:
        raise ValueError("No eligible steps for invented-rule injection")

    if step_number is None:
        target = rng.choice(eligible_steps)
    else:
        matches = [step for step in eligible_steps if step.step_number == step_number]
        if not matches:
            raise ValueError(f"Step {step_number} is not eligible for invented-rule injection")
        target = matches[0]

    invented_rule_used, invented_rule_type = rng.choice(_invented_rule_choices(system))

    mutated_step = replace(
        target,
        rule_used=invented_rule_used,
        rule_type=invented_rule_type,
    )
    mutated_steps = tuple(
        mutated_step if step.step_number == target.step_number else step
        for step in chain.steps
    )
    mutated_chain = DerivationChain(
        starting_expression=chain.starting_expression,
        steps=mutated_steps,
        final_result=chain.final_result,
        seed=chain.seed,
    )

    error_info = ErrorInfo(
        error_type=ErrorType.E_INV,
        step_number=target.step_number,
        correct_result=target.result_symbol,
        injected_result=target.result_symbol,
        correct_after=target.after,
        injected_after=target.after,
        original_chain=chain,
        correct_rule_used=target.rule_used,
        injected_rule_used=invented_rule_used,
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
    """Inject a wrong result and recompute a locally valid but globally wrong suffix."""
    rng = _resolve_rng(rng=rng, seed=seed)

    eligible_steps = get_cascading_eligible_steps(chain)
    if not eligible_steps:
        raise ValueError("No eligible steps for cascading wrong-result injection")

    if step_number is None:
        targets = list(eligible_steps)
        rng.shuffle(targets)
    else:
        matches = [step for step in eligible_steps if step.step_number == step_number]
        if not matches:
            raise ValueError(
                f"Step {step_number} is not eligible for cascading wrong-result injection"
            )
        targets = matches

    selected: tuple[ChainStep, Symbol, Expression, tuple[ChainStep, ...], Symbol] | None = None

    for target in targets:
        assert target.result_symbol is not None
        wrong_choices = [sym for sym in system.symbols if sym != target.result_symbol]
        rng.shuffle(wrong_choices)

        for injected_result in wrong_choices:
            injected_after = _build_injected_after(target, injected_result)
            suffix, final_result = _reduce_from(injected_after, system)
            if final_result != chain.final_result:
                selected = (target, injected_result, injected_after, suffix, final_result)
                break

    if selected is None:
        raise ValueError("Could not inject a cascading wrong result that changes the final result")

    target, injected_result, injected_after, suffix, final_result = selected

    mutated_root = replace(target, result_symbol=injected_result, after=injected_after)

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
        error_type=ErrorType.E_CASC,
        step_number=target.step_number,
        correct_result=target.result_symbol,
        injected_result=injected_result,
        correct_after=target.after,
        injected_after=injected_after,
        original_chain=chain,
    )
    return mutated_chain, error_info
