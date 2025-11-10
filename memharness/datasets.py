from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .data import iter_conversations, split_samples
from .types import ConversationSample


@dataclass
class DatasetConfig:
    root: Optional[Path] = None
    split: str = "eval"
    eval_ratio: float = 0.8
    seed: int = 42
    include: Optional[List[str]] = None
    limit: Optional[int] = None


@dataclass
class DatasetSpec:
    name: str
    description: str
    loader: Callable[[DatasetConfig], List[ConversationSample]]
    default_root: Path = Path("MemData/FirstAgent")
    available_splits: tuple[str, ...] = ("eval", "validation")
    extra: dict = field(default_factory=dict)


def load_membench_first_agent(config: DatasetConfig) -> List[ConversationSample]:
    root = config.root or Path("data/MemBench/FirstAgent")
    include = set(config.include) if config.include else None
    samples = list(iter_conversations(root, include))
    eval_split, validation_split = split_samples(samples, config.eval_ratio, config.seed)
    split_map = {
        "eval": eval_split,
        "validation": validation_split,
        "all": samples,
    }
    selected = split_map.get(config.split, eval_split)
    selected = sorted(selected, key=lambda sample: sample.id)
    if config.limit:
        selected = selected[: config.limit]
    return selected


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "memharness-first-agent": DatasetSpec(
        name="memharness-first-agent",
        description="MemBench participation (first-agent) categorical dataset.",
        loader=load_membench_first_agent,
        default_root=Path("MemData/FirstAgent"),
        available_splits=("eval", "validation", "all"),
    )
}


def list_datasets() -> List[DatasetSpec]:
    return list(DATASET_REGISTRY.values())


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}") from exc


def load_dataset(name: str, config: DatasetConfig) -> List[ConversationSample]:
    spec = get_dataset(name)
    # Use spec default root if config root not provided
    if config.root is None:
        config.root = spec.default_root
    return spec.loader(config)


__all__ = [
    "DatasetConfig",
    "DatasetSpec",
    "DATASET_REGISTRY",
    "get_dataset",
    "list_datasets",
    "load_dataset",
]
