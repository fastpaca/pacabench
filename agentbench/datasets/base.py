from abc import ABC, abstractmethod

from agentbench.config import DatasetConfig
from agentbench.types import Case


class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def load(self, limit: int | None = None) -> list[Case]:
        """Load cases for this dataset, optionally limited."""
        pass
