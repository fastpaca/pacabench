from agentbench.config import DatasetConfig
from agentbench.context import EvalContext
from agentbench.datasets.base import BaseDataset
from agentbench.datasets.git import GitDataset
from agentbench.datasets.hf import HuggingFaceDataset
from agentbench.datasets.local import LocalDataset


def get_dataset(config: DatasetConfig, ctx: EvalContext) -> BaseDataset:
    if config.source.startswith("git:"):
        return GitDataset(config, ctx)
    elif config.source.startswith("huggingface:"):
        return HuggingFaceDataset(config, ctx)
    else:
        return LocalDataset(config, ctx)
