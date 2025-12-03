"""Base runner interface."""

from abc import ABC, abstractmethod

from pacabench.models import AgentConfig, Case, RunnerOutput


class BaseRunner(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        """Start the runner process/resources."""
        pass

    @abstractmethod
    async def run_case(self, case: Case) -> RunnerOutput:
        """Run a single case."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the runner process/resources."""
        pass
