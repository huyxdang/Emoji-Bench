import random

import pytest

from emoji_bench.benchmark_types import ErrorType
from emoji_bench.chain_generator import (
    find_leftmost_innermost,
    generate_chain,
    replace_at_path,
)
from emoji_bench.error_injector import (
    get_cascading_eligible_steps,
    get_wrong_rule_eligible_steps,
    get_wrong_result_eligible_steps,
    inject_cascading_wrong_result,
    inject_wrong_rule,
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


def test_wrong_rule_eligible_steps_require_alternative_rules():
    easy_system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    easy_chain = generate_chain(easy_system, length=3, seed=7)
    assert get_wrong_rule_eligible_steps(easy_chain, easy_system) == ()

    medium_system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    medium_chain = generate_chain(medium_system, length=6, seed=12)
    assert get_wrong_rule_eligible_steps(medium_chain, medium_system)


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


def test_inject_wrong_rule_changes_only_rule_metadata():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    injected_chain, error_info = inject_wrong_rule(chain, system, seed=99)
    changed_steps = [
        (original, injected)
        for original, injected in zip(chain.steps, injected_chain.steps, strict=True)
        if original != injected
    ]

    assert len(changed_steps) == 1
    original_step, injected_step = changed_steps[0]
    assert injected_step.before == original_step.before
    assert injected_step.after == original_step.after
    assert injected_step.result_symbol == original_step.result_symbol
    assert injected_step.rule_used != original_step.rule_used
    assert error_info.original_chain == chain


def test_inject_wrong_rule_records_metadata():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    injected_chain, error_info = inject_wrong_rule(chain, system, seed=99)
    injected_step = injected_chain.steps[error_info.step_number - 1]
    original_step = chain.steps[error_info.step_number - 1]

    assert error_info.error_type is ErrorType.E_RULE
    assert error_info.correct_rule_used == original_step.rule_used
    assert error_info.injected_rule_used == injected_step.rule_used
    assert error_info.correct_rule_used != error_info.injected_rule_used
    assert error_info.correct_result == original_step.result_symbol
    assert error_info.injected_result == original_step.result_symbol
    assert error_info.correct_after == original_step.after
    assert error_info.injected_after == original_step.after


def test_inject_wrong_rule_rejects_ineligible_system():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=3, seed=7)

    with pytest.raises(ValueError, match="No eligible steps"):
        inject_wrong_rule(chain, system, seed=99)


def test_inject_wrong_rule_rejects_seed_and_rng_together():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    with pytest.raises(ValueError, match="either seed or rng"):
        inject_wrong_rule(chain, system, seed=99, rng=random.Random(99))


def test_cascading_eligible_steps_are_non_terminal_result_steps():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    eligible = get_cascading_eligible_steps(chain)

    assert eligible
    assert all(step.result_symbol is not None for step in eligible)
    assert all(step.step_number < chain.steps[-1].step_number for step in eligible)


def test_inject_cascading_wrong_result_changes_suffix_but_not_prefix():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)
    step_number = get_cascading_eligible_steps(chain)[0].step_number

    injected_chain, error_info = inject_cascading_wrong_result(
        chain,
        system,
        step_number=step_number,
        seed=99,
    )

    assert injected_chain.steps[: step_number - 1] == chain.steps[: step_number - 1]
    assert injected_chain.steps[step_number - 1].before == chain.steps[step_number - 1].before
    assert injected_chain.steps[step_number - 1].result_symbol != chain.steps[step_number - 1].result_symbol
    assert injected_chain.steps[step_number - 1 :] != chain.steps[step_number - 1 :]
    assert error_info.original_chain == chain


def test_inject_cascading_wrong_result_recomputes_locally_valid_suffix():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)
    step_number = get_cascading_eligible_steps(chain)[0].step_number

    injected_chain, error_info = inject_cascading_wrong_result(
        chain,
        system,
        step_number=step_number,
        seed=99,
    )

    root_step = injected_chain.steps[step_number - 1]
    assert error_info.error_type is ErrorType.E_CASC
    assert evaluate(root_step.reduced_subexpr, system) == error_info.correct_result
    assert root_step.result_symbol == error_info.injected_result
    assert root_step.result_symbol != evaluate(root_step.reduced_subexpr, system)

    for step in injected_chain.steps[step_number:]:
        if step.result_symbol is not None:
            assert evaluate(step.reduced_subexpr, system) == step.result_symbol


def test_inject_cascading_wrong_result_preserves_outer_expression_shape():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)
    target = next(
        step for step in get_cascading_eligible_steps(chain)
        if not isinstance(step.after, SymbolLiteral)
    )

    injected_chain, error_info = inject_cascading_wrong_result(
        chain,
        system,
        step_number=target.step_number,
        seed=99,
    )

    injected_step = injected_chain.steps[target.step_number - 1]
    path = find_leftmost_innermost(target.before)

    assert path is not None
    assert not isinstance(injected_step.after, SymbolLiteral)
    assert injected_chain.steps[target.step_number].before == injected_step.after
    assert injected_step.after == replace_at_path(
        target.before,
        path,
        SymbolLiteral(error_info.injected_result),
    )


def test_inject_cascading_wrong_result_updates_final_result():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    injected_chain, _ = inject_cascading_wrong_result(chain, system, seed=99)

    assert injected_chain.final_result != chain.final_result
    assert injected_chain.final_result == injected_chain.steps[-1].after.symbol


def test_inject_cascading_wrong_result_rejects_ineligible_step_number():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=6, seed=12)

    with pytest.raises(ValueError, match="not eligible"):
        inject_cascading_wrong_result(
            chain,
            system,
            step_number=chain.steps[-1].step_number,
            seed=99,
        )
