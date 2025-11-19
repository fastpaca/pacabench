from agentbench.config import DatasetConfig
from agentbench.datasets.base import BaseDataset
from agentbench.datasets.git import GitDataset
from agentbench.datasets.hf import HuggingFaceDataset
from agentbench.datasets.local import LocalDataset


def get_dataset(config: DatasetConfig) -> BaseDataset:
    if config.source.startswith("git:"):
        return GitDataset(config)
    elif config.source.startswith("huggingface:"):
        return HuggingFaceDataset(config)
    else:
        return LocalDataset(config)

