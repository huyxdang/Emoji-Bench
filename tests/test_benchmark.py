from emoji_bench.benchmark import generate_benchmark_instance
from emoji_bench.benchmark_types import Condition, ErrorType
from emoji_bench.generator import generate_system
from emoji_bench.interpreter import evaluate
from emoji_bench.prompt_formatter import format_benchmark_prompt


def test_generate_clean_benchmark_instance():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    instance = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.CLEAN,
        chain_seed=12,
        instance_id="clean-1",
    )

    assert instance.condition is Condition.CLEAN
    assert instance.has_error is False
    assert instance.error_info is None
    assert instance.instance_id == "clean-1"
    assert instance.prompt == format_benchmark_prompt(system, instance.chain)
    assert instance.chain.final_result == evaluate(instance.chain.starting_expression, system)


def test_generate_error_benchmark_instance():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    instance = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_seed=99,
        instance_id="err-1",
    )

    assert instance.condition is Condition.ERROR_INJECTED
    assert instance.has_error is True
    assert instance.error_info is not None
    assert instance.error_info.error_type is ErrorType.E_RES
    assert instance.instance_id == "err-1"
    assert instance.prompt == format_benchmark_prompt(system, instance.chain)
    assert instance.chain.steps[:-1] == instance.error_info.original_chain.steps[:-1]
    assert instance.chain.final_result != evaluate(instance.chain.starting_expression, system)


def test_generate_cascading_error_benchmark_instance():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    instance = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_CASC,
        error_seed=99,
        instance_id="casc-1",
    )

    assert instance.condition is Condition.ERROR_INJECTED
    assert instance.has_error is True
    assert instance.error_info is not None
    assert instance.error_info.error_type is ErrorType.E_CASC
    assert instance.instance_id == "casc-1"
    assert instance.prompt == format_benchmark_prompt(system, instance.chain)
    assert instance.chain.final_result != evaluate(instance.chain.starting_expression, system)
    assert instance.chain.steps[:-1] != instance.error_info.original_chain.steps[:-1]


def test_generate_wrong_rule_benchmark_instance():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    instance = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_RULE,
        error_seed=99,
        instance_id="rule-1",
    )

    assert instance.condition is Condition.ERROR_INJECTED
    assert instance.has_error is True
    assert instance.error_info is not None
    assert instance.error_info.error_type is ErrorType.E_RULE
    assert instance.instance_id == "rule-1"
    assert instance.prompt == format_benchmark_prompt(system, instance.chain)
    assert instance.chain.final_result == evaluate(instance.chain.starting_expression, system)
    assert instance.error_info.correct_rule_used != instance.error_info.injected_rule_used


def test_generate_error_benchmark_instance_is_deterministic():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    i1 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_seed=99,
    )
    i2 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_seed=99,
    )

    assert i1.chain == i2.chain
    assert i1.error_info == i2.error_info
    assert i1.prompt == i2.prompt


def test_generate_error_benchmark_instance_defaults_error_seed():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    i1 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
    )
    i2 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_seed=13,
    )

    assert i1.chain == i2.chain
    assert i1.error_info == i2.error_info


def test_generate_cascading_error_benchmark_instance_is_deterministic():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    i1 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_CASC,
        error_seed=99,
    )
    i2 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_CASC,
        error_seed=99,
    )

    assert i1.chain == i2.chain
    assert i1.error_info == i2.error_info
    assert i1.prompt == i2.prompt


def test_generate_wrong_rule_benchmark_instance_is_deterministic():
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    i1 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_RULE,
        error_seed=99,
    )
    i2 = generate_benchmark_instance(
        system,
        length=6,
        condition=Condition.ERROR_INJECTED,
        chain_seed=12,
        error_type=ErrorType.E_RULE,
        error_seed=99,
    )

    assert i1.chain == i2.chain
    assert i1.error_info == i2.error_info
    assert i1.prompt == i2.prompt
