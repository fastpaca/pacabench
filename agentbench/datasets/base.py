"""Base dataset abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel, Field

from agentbench.stages.case import Case
from agentbench.stages.result import CaseResult


class Dataset(BaseModel, ABC):
    """Abstract base class for datasets."""

    split: str | None = Field(None, description="Dataset split if applicable")

    model_config = {"extra": "forbid"}

    @abstractmethod
    async def load(self, limit: int | None = None) -> Iterable[Case]:
        """
        Load cases from the dataset.

        Args:
            limit: Optional limit on number of cases

        Returns:
            Iterable of Case objects
        """
        ...

    @abstractmethod
    async def eval(
        self,
        case: Case,
        result: CaseResult,
        judge_model: str = "gpt-4o-mini",
        judge_client=None,
    ) -> CaseResult:
        """
        Evaluate a case result and complete it with evaluation fields.

        Args:
            case: Test case
            result: Partial CaseResult (with output, error, duration, llm_metrics)
            judge_model: Judge model name
            judge_client: OpenAI client for judge (optional)

        Returns:
            Complete CaseResult with evaluation fields (passed, f1_score, etc.)
        """
        ...
