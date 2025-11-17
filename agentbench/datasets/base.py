"""Base dataset abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel, Field

from agentbench.types import Case


class EvaluationResult(BaseModel):
    """Results from dataset evaluation."""

    passed: bool = Field(..., description="Whether the case passed")
    f1_score: float | None = Field(None, description="F1 score if applicable")
    f1_passed: bool | None = Field(None, description="Whether F1 evaluation passed")
    judge_passed: bool | None = Field(None, description="Whether judge evaluation passed")

    model_config = {"extra": "forbid"}


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
        output: str | None,
        error: str | None,
        judge_model: str = "gpt-4o-mini",
        judge_client=None,
    ) -> tuple[EvaluationResult, dict[str, int] | None]:
        """
        Evaluate a model output against a test case.

        Args:
            case: Test case
            output: Model output (None if error occurred)
            error: Error message if any
            judge_model: Judge model name
            judge_client: OpenAI client for judge (optional)

        Returns:
            Tuple of (EvaluationResult, judge_metrics dict or None)
            Judge metrics are operational data, not part of evaluation result
        """
        ...
