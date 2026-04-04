from __future__ import annotations

import random

from emoji_bench.benchmark_types import BenchmarkInstance, ErrorInfo
from emoji_bench.chain_types import ChainStep, DerivationChain
from emoji_bench.expressions import BinaryOp, Expression, SymbolLiteral, UnaryTransform
from emoji_bench.prompt_formatter import format_benchmark_prompt
from emoji_bench.types import FormalSystem, OperationTable, Symbol, TransformationRule

TWO_DIGIT_LABELS: tuple[str, ...] = tuple(f"{value:02d}" for value in range(10, 100))


def _resolve_rng(
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> random.Random:
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both")
    if rng is not None:
        return rng
    return random.Random(seed)


def build_two_digit_symbol_map(
    symbols: tuple[Symbol, ...],
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> dict[Symbol, Symbol]:
    """Assign each symbol a unique random two-digit label."""
    if len(symbols) > len(TWO_DIGIT_LABELS):
        raise ValueError("Not enough two-digit labels to relabel all symbols")

    resolved_rng = _resolve_rng(seed=seed, rng=rng)
    labels = list(TWO_DIGIT_LABELS)
    resolved_rng.shuffle(labels)
    selected = labels[: len(symbols)]
    return {
        old_symbol: Symbol(label)
        for old_symbol, label in zip(symbols, selected, strict=True)
    }


def relabel_expression(
    expr: Expression,
    symbol_map: dict[Symbol, Symbol],
) -> Expression:
    match expr:
        case SymbolLiteral(symbol):
            return SymbolLiteral(symbol_map[symbol])
        case BinaryOp(op_name, left, right):
            return BinaryOp(
                op_name,
                relabel_expression(left, symbol_map),
                relabel_expression(right, symbol_map),
            )
        case UnaryTransform(transform_name, operand):
            return UnaryTransform(transform_name, relabel_expression(operand, symbol_map))

    raise ValueError(f"Unknown expression type: {type(expr)}")


def relabel_text(
    text: str,
    symbol_map: dict[Symbol, Symbol],
) -> str:
    """Relabel raw prompt text using the provided symbol mapping."""
    relabeled = text
    replacements = sorted(
        symbol_map.items(),
        key=lambda item: len(item[0].emoji),
        reverse=True,
    )
    for source, target in replacements:
        relabeled = relabeled.replace(source.emoji, target.emoji)
    return relabeled


def relabel_chain(
    chain: DerivationChain,
    symbol_map: dict[Symbol, Symbol],
) -> DerivationChain:
    return DerivationChain(
        starting_expression=relabel_expression(chain.starting_expression, symbol_map),
        steps=tuple(_relabel_step(step, symbol_map) for step in chain.steps),
        final_result=symbol_map[chain.final_result],
        seed=chain.seed,
    )


def relabel_system(
    system: FormalSystem,
    symbol_map: dict[Symbol, Symbol],
) -> FormalSystem:
    relabeled_symbols = tuple(symbol_map[symbol] for symbol in system.symbols)

    base_operations = tuple(
        OperationTable(
            name=op.name,
            symbol_id=op.symbol_id,
            symbols=relabeled_symbols,
            table={
                (symbol_map[left], symbol_map[right]): symbol_map[result]
                for (left, right), result in op.table.items()
            },
        )
        for op in system.base_operations
    )
    transformations = tuple(
        TransformationRule(
            name=transform.name,
            mapping={
                symbol_map[source]: symbol_map[target]
                for source, target in transform.mapping.items()
            },
            distributes_over=transform.distributes_over,
        )
        for transform in system.transformations
    )

    return FormalSystem(
        name=system.name,
        seed=system.seed,
        symbols=relabeled_symbols,
        base_operations=base_operations,
        derived_operations=system.derived_operations,
        transformations=transformations,
    )


def relabel_benchmark_instance(
    instance: BenchmarkInstance,
    symbol_map: dict[Symbol, Symbol],
) -> BenchmarkInstance:
    relabeled_system = relabel_system(instance.system, symbol_map)
    relabeled_chain = relabel_chain(instance.chain, symbol_map)
    relabeled_error_info = _relabel_error_info(instance.error_info, symbol_map)
    prompt = format_benchmark_prompt(relabeled_system, relabeled_chain)

    return BenchmarkInstance(
        system=relabeled_system,
        chain=relabeled_chain,
        condition=instance.condition,
        has_error=instance.has_error,
        prompt=prompt,
        error_info=relabeled_error_info,
        instance_id=instance.instance_id,
    )


def relabel_system_with_two_digit_numbers(
    system: FormalSystem,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> tuple[FormalSystem, dict[Symbol, Symbol]]:
    symbol_map = build_two_digit_symbol_map(system.symbols, seed=seed, rng=rng)
    return relabel_system(system, symbol_map), symbol_map


def relabel_benchmark_instance_with_two_digit_numbers(
    instance: BenchmarkInstance,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> BenchmarkInstance:
    symbol_map = build_two_digit_symbol_map(instance.system.symbols, seed=seed, rng=rng)
    return relabel_benchmark_instance(instance, symbol_map)


def _relabel_step(step: ChainStep, symbol_map: dict[Symbol, Symbol]) -> ChainStep:
    return ChainStep(
        step_number=step.step_number,
        before=relabel_expression(step.before, symbol_map),
        reduced_subexpr=relabel_expression(step.reduced_subexpr, symbol_map),
        result_symbol=(
            symbol_map[step.result_symbol] if step.result_symbol is not None else None
        ),
        after=relabel_expression(step.after, symbol_map),
        rule_used=step.rule_used,
        rule_type=step.rule_type,
        expanded_to=(
            relabel_expression(step.expanded_to, symbol_map)
            if step.expanded_to is not None
            else None
        ),
    )


def _relabel_error_info(
    error_info: ErrorInfo | None,
    symbol_map: dict[Symbol, Symbol],
) -> ErrorInfo | None:
    if error_info is None:
        return None

    return ErrorInfo(
        error_type=error_info.error_type,
        step_number=error_info.step_number,
        correct_result=(
            symbol_map[error_info.correct_result]
            if error_info.correct_result is not None
            else None
        ),
        injected_result=(
            symbol_map[error_info.injected_result]
            if error_info.injected_result is not None
            else None
        ),
        correct_after=relabel_expression(error_info.correct_after, symbol_map),
        injected_after=relabel_expression(error_info.injected_after, symbol_map),
        original_chain=relabel_chain(error_info.original_chain, symbol_map),
        correct_rule_used=error_info.correct_rule_used,
        injected_rule_used=error_info.injected_rule_used,
    )
