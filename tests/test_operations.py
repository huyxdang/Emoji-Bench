import random

from emoji_bench.operations import generate_operation_table
from emoji_bench.symbols import sample_symbols


def _make_symbols(n=3, seed=42):
    return sample_symbols(n, random.Random(seed))


def test_table_completeness():
    syms = _make_symbols(4)
    rng = random.Random(99)
    op = generate_operation_table(syms, rng, "test", "⊕")
    for a in syms:
        for b in syms:
            assert (a, b) in op.table


def test_table_closure():
    syms = _make_symbols(4)
    rng = random.Random(99)
    op = generate_operation_table(syms, rng, "test", "⊕")
    sym_set = set(syms)
    for result in op.table.values():
        assert result in sym_set


def test_table_deterministic():
    syms = _make_symbols(3)
    op1 = generate_operation_table(syms, random.Random(77), "t", "⊕")
    op2 = generate_operation_table(syms, random.Random(77), "t", "⊕")
    assert op1.table == op2.table


def test_commutative():
    syms = _make_symbols(4)
    rng = random.Random(55)
    op = generate_operation_table(syms, rng, "comm", "⊕", commutative=True)
    for a in syms:
        for b in syms:
            assert op.table[(a, b)] == op.table[(b, a)]
