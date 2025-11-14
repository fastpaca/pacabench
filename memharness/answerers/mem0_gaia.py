"""Mem0-powered GAIA answerer for agentic workflows with memory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from memharness.executors.mem0_gaia import Mem0GAIAExecutor


def mem0_gaia_answerer(
    model: str,
    provider: str,
    max_steps: int = 15,
    planning_interval: int | None = None,
    mem0_api_key: str | None = None,
    mem0_config: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Create a mem0-powered GAIA answerer with lazy-initialized executor.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
        provider: Provider name (e.g., "openai", "anthropic")
        max_steps: Maximum agent steps (default: 15)
        planning_interval: Steps between replanning (None = no replanning)
        mem0_api_key: Optional API key for mem0
        mem0_config: Optional mem0 configuration dict

    Returns:
        Async task function compatible with pydantic-evals
    """
    _executor = None

    def get_executor() -> Mem0GAIAExecutor:
        nonlocal _executor
        if _executor is not None:
            return _executor

        _executor = Mem0GAIAExecutor(
            model=model,
            provider=provider,
            max_steps=max_steps,
            planning_interval=planning_interval,
            mem0_api_key=mem0_api_key,
            mem0_config=mem0_config,
        )
        return _executor

    async def task(inputs: dict[str, Any]) -> str:
        executor = get_executor()
        result = await executor.execute(inputs)
        return result

    return task
