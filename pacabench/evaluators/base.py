"""Base evaluator interface."""

from abc import ABC, abstractmethod

from pacabench.models import Case, EvaluationResult, EvaluatorConfig, RunnerOutput


class BaseEvaluator(ABC):
    def __init__(self, config: EvaluatorConfig):
        self.config = config

    @abstractmethod
    async def evaluate(
        self, case: Case, output: RunnerOutput, proxy_url: str | None = None
    ) -> EvaluationResult:
        pass
