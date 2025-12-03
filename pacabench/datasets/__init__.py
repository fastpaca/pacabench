"""Dataset loaders for PacaBench."""

from pathlib import Path

from pacabench.datasets.base import BaseDataset
from pacabench.datasets.git import GitDataset
from pacabench.datasets.hf import HuggingFaceDataset
from pacabench.datasets.local import LocalDataset
from pacabench.models import DatasetConfig


def get_dataset(
    config: DatasetConfig,
    root_dir: Path,
    datasets_cache_dir: Path,
    env: dict[str, str] | None = None,
) -> BaseDataset:
    """Create a dataset loader based on the source type."""
    kwargs = {
        "config": config,
        "root_dir": root_dir,
        "datasets_cache_dir": datasets_cache_dir,
        "env": env,
    }
    if config.source.startswith("git:"):
        return GitDataset(**kwargs)
    elif config.source.startswith("huggingface:"):
        return HuggingFaceDataset(**kwargs)
    else:
        return LocalDataset(**kwargs)
