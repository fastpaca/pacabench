"""Abstract base class for task executors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Executor(ABC):
    """Abstract base for all task executors.

    Executors implement the logic for running tasks on different types of datasets.
    Each executor can implement its own memory strategy, prompting approach, and
    execution flow without being constrained by a shared abstraction.
    """

    def __init__(self, model: str, provider: str, **kwargs):
        """Initialize executor with model configuration.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5")
            provider: Provider name ("openai", "anthropic")
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self.provider = provider
        self.kwargs = kwargs

    @abstractmethod
    async def execute(self, inputs: dict) -> str:
        """Execute task and return result.

        Args:
            inputs: Task inputs (format varies by dataset type)

        Returns:
            Task output as string
        """
        pass
