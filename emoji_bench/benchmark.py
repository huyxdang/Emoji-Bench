from __future__ import annotations

from emoji_bench.benchmark_types import BenchmarkInstance, Condition, ErrorType
from emoji_bench.chain_generator import generate_chain
from emoji_bench.error_injector import (
    inject_cascading_wrong_result,
    inject_invented_rule,
    inject_wrong_result,
    inject_wrong_rule,
)
from emoji_bench.prompt_formatter import format_benchmark_prompt
from emoji_bench.types import FormalSystem


def generate_benchmark_instance(
    system: FormalSystem,
    *,
    length: int,
    condition: Condition,
    chain_seed: int,
    error_type: ErrorType = ErrorType.E_RES,
    error_seed: int | None = None,
    instance_id: str | None = None,
) -> BenchmarkInstance:
    """Generate a clean or error-injected benchmark instance."""
    chain = generate_chain(system, length=length, seed=chain_seed)

    if condition is Condition.CLEAN:
        prompt = format_benchmark_prompt(system, chain)
        return BenchmarkInstance(
            system=system,
            chain=chain,
            condition=condition,
            has_error=False,
            prompt=prompt,
            error_info=None,
            instance_id=instance_id,
        )

    if condition is Condition.ERROR_INJECTED:
        if error_seed is None:
            error_seed = chain_seed + 1
        if error_type is ErrorType.E_RES:
            injected_chain, error_info = inject_wrong_result(
                chain,
                system,
                seed=error_seed,
            )
        elif error_type is ErrorType.E_RULE:
            injected_chain, error_info = inject_wrong_rule(
                chain,
                system,
                seed=error_seed,
            )
        elif error_type is ErrorType.E_INV:
            injected_chain, error_info = inject_invented_rule(
                chain,
                system,
                seed=error_seed,
            )
        elif error_type is ErrorType.E_CASC:
            injected_chain, error_info = inject_cascading_wrong_result(
                chain,
                system,
                seed=error_seed,
            )
        else:
            raise ValueError(f"Unsupported error type: {error_type}")
        prompt = format_benchmark_prompt(system, injected_chain)
        return BenchmarkInstance(
            system=system,
            chain=injected_chain,
            condition=condition,
            has_error=True,
            prompt=prompt,
            error_info=error_info,
            instance_id=instance_id,
        )

    raise ValueError(f"Unsupported condition: {condition}")
