from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from emoji_bench.chain_types import DerivationChain
from emoji_bench.expressions import Expression
from emoji_bench.types import FormalSystem, Symbol


class Condition(str, Enum):
    CLEAN = "clean"
    ERROR_INJECTED = "error_injected"


class ErrorType(str, Enum):
    E_RES = "E-RES"


@dataclass(frozen=True)
class ErrorInfo:
    error_type: ErrorType
    step_number: int
    correct_result: Symbol
    injected_result: Symbol
    correct_after: Expression
    injected_after: Expression
    original_chain: DerivationChain


@dataclass(frozen=True)
class BenchmarkInstance:
    system: FormalSystem
    chain: DerivationChain
    condition: Condition
    has_error: bool
    prompt: str
    error_info: ErrorInfo | None = None
    instance_id: str | None = None
