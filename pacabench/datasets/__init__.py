from pacabench.config import DatasetConfig
from pacabench.context import EvalContext
from pacabench.datasets.base import BaseDataset
from pacabench.datasets.git import GitDataset
from pacabench.datasets.hf import HuggingFaceDataset
from pacabench.datasets.local import LocalDataset


def get_dataset(config: DatasetConfig, ctx: EvalContext) -> BaseDataset:
    if config.source.startswith("git:"):
        return GitDataset(config, ctx)
    elif config.source.startswith("huggingface:"):
        return HuggingFaceDataset(config, ctx)
    else:
        return LocalDataset(config, ctx)
