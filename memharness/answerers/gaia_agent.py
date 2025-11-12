"""GAIA answerer factory for pydantic-evals compatibility."""

from collections.abc import Awaitable, Callable
from typing import Any

from memharness.executors.gaia import GAIAExecutor


def gaia_answerer(
    model: str,
    provider: str,
    max_steps: int = 15,
    planning_interval: int | None = None,
) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Create a GAIA answerer task function with lazy-initialized executor.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
        provider: Provider name (e.g., "openai", "anthropic")
        max_steps: Maximum agent steps (default: 15)
        planning_interval: Steps between replanning (None = no replanning)

    Returns:
        Async task function compatible with pydantic-evals
    """
    _executor = None

    def get_executor() -> GAIAExecutor:
        nonlocal _executor
        if _executor is not None:
            return _executor

        _executor = GAIAExecutor(
            model=model,
            provider=provider,
            max_steps=max_steps,
            planning_interval=planning_interval,
        )
        return _executor

    async def task(inputs: dict[str, Any]) -> str:
        executor = get_executor()
        result = await executor.execute(inputs)
        return result

    return task
