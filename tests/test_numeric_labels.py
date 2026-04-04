import pytest

from emoji_bench.benchmark import generate_benchmark_instance
from emoji_bench.benchmark_types import Condition, ErrorType
from emoji_bench.chain_generator import generate_chain
from emoji_bench.generator import generate_system
from emoji_bench.interpreter import evaluate
from emoji_bench.numeric_labels import (
    build_two_digit_symbol_map,
    relabel_benchmark_instance_with_two_digit_numbers,
    relabel_chain,
    relabel_system_with_two_digit_numbers,
)
from emoji_bench.prompt_formatter import format_benchmark_prompt


def test_build_two_digit_symbol_map_is_deterministic_and_unique():
    system = generate_system(n_symbols=4, random_seed=77)

    mapping_a = build_two_digit_symbol_map(system.symbols, seed=123)
    mapping_b = build_two_digit_symbol_map(system.symbols, seed=123)

    assert mapping_a == mapping_b
    labels = [symbol.emoji for symbol in mapping_a.values()]
    assert len(set(labels)) == len(labels)
    assert all(label.isdigit() and len(label) == 2 for label in labels)


def test_relabel_system_and_chain_with_two_digit_numbers_preserve_evaluation():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)
    relabeled_system, symbol_map = relabel_system_with_two_digit_numbers(system, seed=123)
    relabeled_chain = relabel_chain(chain, symbol_map)

    assert all(symbol.emoji.isdigit() and len(symbol.emoji) == 2 for symbol in relabeled_system.symbols)
    assert all(not symbol.emoji.isdigit() for symbol in system.symbols)
    assert evaluate(relabeled_chain.starting_expression, relabeled_system) == relabeled_chain.final_result


def test_relabel_benchmark_instance_with_two_digit_numbers_preserves_error_case():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    instance = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_RES,
        error_seed=99,
        instance_id="err-1",
    )

    relabeled = relabel_benchmark_instance_with_two_digit_numbers(instance, seed=123)

    assert relabeled.condition is instance.condition
    assert relabeled.has_error is True
    assert relabeled.instance_id == "err-1"
    assert relabeled.error_info is not None
    assert relabeled.error_info.error_type is ErrorType.E_RES
    assert relabeled.prompt == format_benchmark_prompt(relabeled.system, relabeled.chain)
    assert relabeled.prompt != instance.prompt
    assert all(symbol.emoji.isdigit() and len(symbol.emoji) == 2 for symbol in relabeled.system.symbols)
    assert relabeled.chain.final_result != evaluate(relabeled.chain.starting_expression, relabeled.system)


def test_build_two_digit_symbol_map_rejects_seed_and_rng_together():
    system = generate_system(n_symbols=4, random_seed=77)

    with pytest.raises(ValueError, match="either seed or rng"):
        build_two_digit_symbol_map(system.symbols, seed=123, rng=object())
