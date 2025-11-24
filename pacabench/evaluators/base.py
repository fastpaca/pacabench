from abc import ABC, abstractmethod

from pacabench.config import EvaluatorConfig
from pacabench.types import Case, EvaluationResult, RunnerOutput


class BaseEvaluator(ABC):
    def __init__(self, config: EvaluatorConfig):
        self.config = config

    @abstractmethod
    async def evaluate(
        self, case: Case, output: RunnerOutput, proxy_url: str | None = None
    ) -> EvaluationResult:
        pass
