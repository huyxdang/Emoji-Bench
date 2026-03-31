import random

from emoji_bench.chain_generator import generate_chain, reduce_expression
from emoji_bench.expressions import BinaryOp, SymbolLiteral
from emoji_bench.generator import generate_system
from emoji_bench.prompt_formatter import format_benchmark_prompt, format_chain


def test_prompt_has_all_sections():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=5, rng=random.Random(7))
    prompt = format_benchmark_prompt(system, chain)
    assert "=== RULES ===" in prompt
    assert "=== DERIVATION ===" in prompt
    assert "=== TASK ===" in prompt


def test_prompt_contains_system_name():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=5, rng=random.Random(7))
    prompt = format_benchmark_prompt(system, chain)
    assert system.name in prompt


def test_prompt_shows_starting_expression():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=5, rng=random.Random(7))
    prompt = format_benchmark_prompt(system, chain)
    assert "Start:" in prompt


def test_prompt_contains_result():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=5, rng=random.Random(7))
    prompt = format_benchmark_prompt(system, chain)
    assert f"Result: {chain.final_result.emoji}" in prompt


def test_prompt_no_internal_names():
    """No internal names like 'op0' or 'dop0' should appear in the prompt."""
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=7, rng=random.Random(42))
    prompt = format_benchmark_prompt(system, chain)
    assert "op0" not in prompt
    assert "dop0" not in prompt


def test_prompt_contains_step_numbers():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=5, rng=random.Random(7))
    prompt = format_benchmark_prompt(system, chain)
    for step in chain.steps:
        assert f"Step {step.step_number}:" in prompt


def test_format_chain_derived_op_shows_expansion():
    """Derived op steps should show the expansion, not a symbol result."""
    system = generate_system(
        n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=77
    )
    chain = generate_chain(system, length=7, rng=random.Random(42))
    text = format_chain(chain, system)
    # Should contain "by definition of ⊗" for expansion steps
    dop = system.derived_operations[0]
    has_expansion = any(s.rule_type == "derived_op" for s in chain.steps)
    if has_expansion:
        assert f"definition of {dop.symbol_id}" in text


def test_format_chain_shows_full_expression_rewrites():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    op_name = system.base_operations[0].name
    a, b = system.symbols[:2]
    repeated = BinaryOp(op_name, SymbolLiteral(a), SymbolLiteral(b))
    expr = BinaryOp(op_name, repeated, repeated)

    chain = reduce_expression(expr, system)
    text = format_chain(
        type("ChainLike", (), {
            "starting_expression": expr,
            "steps": chain,
            "final_result": chain[-1].after.symbol,
        })(),
        system,
    )
    lines = [line for line in text.splitlines() if line.startswith("Step ")]

    assert len(lines) >= 2
    assert lines[0] != lines[1]
    assert " = " in lines[0]
    assert " = " in lines[1]


def test_prompt_task_section_content():
    system = generate_system(n_symbols=3, n_base_ops=1, random_seed=42)
    chain = generate_chain(system, length=3, rng=random.Random(1))
    prompt = format_benchmark_prompt(system, chain)
    assert "Verify whether each step" in prompt
    assert "If all steps are correct" in prompt
