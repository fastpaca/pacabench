from abc import ABC, abstractmethod

from agentbench.config import EvaluatorConfig
from agentbench.types import Case, EvaluationResult, RunnerOutput


class BaseEvaluator(ABC):
    def __init__(self, config: EvaluatorConfig):
        self.config = config

    @abstractmethod
    async def evaluate(self, case: Case, output: RunnerOutput) -> EvaluationResult:
        pass
