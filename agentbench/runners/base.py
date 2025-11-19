from abc import ABC, abstractmethod

from agentbench.config import AgentConfig
from agentbench.types import Case, RunnerOutput


class BaseRunner(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def start(self):
        """Start the runner process/resources."""
        pass

    @abstractmethod
    async def run_case(self, case: Case) -> RunnerOutput:
        """Run a single case."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the runner process/resources."""
        pass

