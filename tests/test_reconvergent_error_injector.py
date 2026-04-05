import random

import pytest

from emoji_bench.benchmark_types import ErrorType
from emoji_bench.chain_generator import generate_chain
from emoji_bench.generator import generate_system
from emoji_bench.interpreter import evaluate
from emoji_bench.reconvergent_error_injector import (
    get_reconvergent_eligible_steps,
    inject_reconvergent_wrong_result,
)


def test_reconvergent_eligible_steps_are_non_terminal_result_steps():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    eligible = get_reconvergent_eligible_steps(chain)

    assert eligible
    assert all(step.result_symbol is not None for step in eligible)
    assert all(step.step_number < chain.steps[-1].step_number for step in eligible)


def test_inject_reconvergent_wrong_result_preserves_final_result_and_root_error():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=1
    )
    chain = generate_chain(system, length=6, seed=1)

    injected_chain, error_info = inject_reconvergent_wrong_result(chain, system, seed=99)

    root_step = injected_chain.steps[error_info.step_number - 1]
    assert error_info.error_type is ErrorType.E_RECONV
    assert root_step.result_symbol == error_info.injected_result
    assert error_info.injected_result != error_info.correct_result
    assert evaluate(root_step.reduced_subexpr, system) == error_info.correct_result
    assert root_step.result_symbol != evaluate(root_step.reduced_subexpr, system)
    assert injected_chain.final_result == chain.final_result
    assert injected_chain.final_result == evaluate(chain.starting_expression, system)

    for step in injected_chain.steps[error_info.step_number:]:
        if step.result_symbol is not None:
            assert evaluate(step.reduced_subexpr, system) == step.result_symbol


def test_inject_reconvergent_wrong_result_prefers_early_step_when_available():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=1
    )
    chain = generate_chain(system, length=6, seed=1)

    _, error_info = inject_reconvergent_wrong_result(chain, system, seed=99)

    assert error_info.step_number == 2


def test_inject_reconvergent_wrong_result_rejects_ineligible_step_number():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=1
    )
    chain = generate_chain(system, length=6, seed=1)

    with pytest.raises(ValueError, match="not eligible"):
        inject_reconvergent_wrong_result(
            chain,
            system,
            step_number=chain.steps[-1].step_number,
            seed=99,
        )


def test_inject_reconvergent_wrong_result_rejects_seed_and_rng_together():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=1
    )
    chain = generate_chain(system, length=6, seed=1)

    with pytest.raises(ValueError, match="either seed or rng"):
        inject_reconvergent_wrong_result(chain, system, seed=99, rng=random.Random(99))
