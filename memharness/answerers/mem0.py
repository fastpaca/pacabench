"""Mem0-powered answerer for Q&A tasks with memory augmentation."""

from __future__ import annotations

from typing import Any

from memharness.executors.mem0_qa import Mem0QAExecutor


def mem0_answerer(
    model: str,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    mem0_api_key: str | None = None,
    mem0_config: dict[str, Any] | None = None,
):
    """Factory that returns async task callable with mem0 memory layer."""
    _executor: Mem0QAExecutor | None = None

    def get_executor() -> Mem0QAExecutor:
        nonlocal _executor
        if _executor is not None:
            return _executor

        _executor = Mem0QAExecutor(
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            mem0_api_key=mem0_api_key,
            mem0_config=mem0_config,
        )
        return _executor

    async def task(inputs: dict) -> str:
        executor = get_executor()
        return await executor.execute(inputs)

    return task
