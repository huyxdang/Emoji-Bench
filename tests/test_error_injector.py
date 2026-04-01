import random

import pytest

from emoji_bench.benchmark_types import ErrorType
from emoji_bench.chain_generator import generate_chain
from emoji_bench.error_injector import (
    get_wrong_result_eligible_steps,
    inject_wrong_result,
)
from emoji_bench.expressions import SymbolLiteral
from emoji_bench.generator import generate_system
from emoji_bench.interpreter import evaluate


def test_wrong_result_eligible_steps_only_terminal_result_step():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    eligible = get_wrong_result_eligible_steps(chain)

    assert eligible == (chain.steps[-1],)


def test_inject_wrong_result_changes_only_last_step_and_final_result():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    injected_chain, error_info = inject_wrong_result(chain, system, seed=99)

    assert injected_chain.steps[:-1] == chain.steps[:-1]
    assert injected_chain.steps[-1].before == chain.steps[-1].before
    assert injected_chain.steps[-1].rule_used == chain.steps[-1].rule_used
    assert injected_chain.steps[-1].result_symbol != chain.steps[-1].result_symbol
    assert injected_chain.final_result == injected_chain.steps[-1].result_symbol
    assert error_info.original_chain == chain


def test_injected_wrong_result_is_actually_wrong():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    injected_chain, error_info = inject_wrong_result(chain, system, seed=99)
    injected_step = injected_chain.steps[-1]

    assert injected_step.result_symbol == error_info.injected_result
    assert evaluate(injected_step.before, system) == error_info.correct_result
    assert injected_step.result_symbol != evaluate(injected_step.before, system)
    assert isinstance(injected_step.after, SymbolLiteral)
    assert injected_step.after.symbol == error_info.injected_result


def test_inject_wrong_result_records_metadata():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    _, error_info = inject_wrong_result(chain, system, seed=99)

    assert error_info.error_type is ErrorType.E_RES
    assert error_info.step_number == chain.steps[-1].step_number
    assert error_info.correct_result == chain.steps[-1].result_symbol
    assert error_info.correct_after == chain.steps[-1].after
    assert error_info.injected_result != error_info.correct_result


def test_inject_wrong_result_rejects_ineligible_step_number():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    with pytest.raises(ValueError, match="not eligible"):
        inject_wrong_result(chain, system, step_number=1, seed=99)


def test_inject_wrong_result_rejects_seed_and_rng_together():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    with pytest.raises(ValueError, match="either seed or rng"):
        inject_wrong_result(chain, system, seed=99, rng=random.Random(99))
