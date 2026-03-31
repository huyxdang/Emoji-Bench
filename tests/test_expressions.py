import random

from emoji_bench.expressions import (
    BinaryOp,
    SymbolLiteral,
    UnaryTransform,
    expr_to_str,
    random_expression,
)
from emoji_bench.generator import generate_system
from emoji_bench.types import Symbol


def test_expr_to_str_literal():
    s = Symbol("🦩")
    assert expr_to_str(SymbolLiteral(s)) == "🦩"


def test_expr_to_str_binary():
    a, b = Symbol("🦩"), Symbol("🧲")
    expr = BinaryOp("⊕", SymbolLiteral(a), SymbolLiteral(b))
    assert expr_to_str(expr) == "(🦩 ⊕ 🧲)"


def test_expr_to_str_nested():
    a, b, c = Symbol("🦩"), Symbol("🧲"), Symbol("🪣")
    inner = BinaryOp("⊕", SymbolLiteral(a), SymbolLiteral(b))
    outer = BinaryOp("⊗", inner, SymbolLiteral(c))
    assert expr_to_str(outer) == "((🦩 ⊕ 🧲) ⊗ 🪣)"


def test_expr_to_str_transform():
    s = Symbol("🦩")
    expr = UnaryTransform("inv", SymbolLiteral(s))
    assert expr_to_str(expr) == "inv(🦩)"


def test_expr_to_str_transform_nested():
    a, b = Symbol("🦩"), Symbol("🧲")
    inner = BinaryOp("⊕", SymbolLiteral(a), SymbolLiteral(b))
    expr = UnaryTransform("inv", inner)
    assert expr_to_str(expr) == "inv((🦩 ⊕ 🧲))"


def test_random_expression_depth_0():
    system = generate_system(n_symbols=3, random_seed=42)
    expr = random_expression(system, depth=0, rng=random.Random(1))
    assert isinstance(expr, SymbolLiteral)


def test_random_expression_produces_valid():
    """random_expression should only reference ops/transforms that exist in the system."""
    system = generate_system(n_symbols=4, n_base_ops=1, n_derived_ops=1, n_transformations=1, random_seed=99)
    rng = random.Random(7)
    op_names = {op.name for op in system.base_operations} | {op.name for op in system.derived_operations}
    tr_names = {t.name for t in system.transformations}

    for _ in range(20):
        expr = random_expression(system, depth=3, rng=rng)
        _check_names(expr, op_names, tr_names, set(system.symbols))


def _check_names(expr, op_names, tr_names, symbols):
    match expr:
        case SymbolLiteral(s):
            assert s in symbols
        case BinaryOp(op_name, left, right):
            assert op_name in op_names
            _check_names(left, op_names, tr_names, symbols)
            _check_names(right, op_names, tr_names, symbols)
        case UnaryTransform(name, operand):
            assert name in tr_names
            _check_names(operand, op_names, tr_names, symbols)
