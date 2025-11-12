"""Abstract base class for benchmark datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic_evals import Dataset

if TYPE_CHECKING:
    from memharness.executors.base import Executor


class BenchmarkDataset(ABC):
    """Abstract base for benchmark datasets.

    Each dataset class knows how to load its data and which executor type
    it should use by default. Datasets can be configured with different
    executor classes to test different memory systems.
    """

    name: str
    default_executor: type[Executor]

    def __init__(self, executor_class: type[Executor] | None = None, **kwargs):
        """Initialize dataset with optional executor override.

        Args:
            executor_class: Optional executor class to use instead of default
            **kwargs: Dataset-specific configuration
        """
        self.executor_class = executor_class or self.default_executor
        self.config = kwargs

    @abstractmethod
    def load(self, limit: int | None = None) -> Dataset:
        """Load and return pydantic-evals Dataset.

        Args:
            limit: Optional limit on number of cases to load

        Returns:
            pydantic-evals Dataset with cases and evaluators
        """
        pass
