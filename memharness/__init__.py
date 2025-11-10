"""MemBench utility library for dataset prep and evaluation."""

from .types import AnswerResult, ConversationSample, QAInput, UsageStats
from .data import dump_jsonl, iter_conversations, load_jsonl, split_samples
from .datasets import DatasetConfig, DatasetSpec, get_dataset, list_datasets, load_dataset
from .evaluation import evaluate_samples
from .answerers import (
    Answerer,
    PydanticAgentAnswerer,
    default_prompt_renderer,
    build_answerer,
    list_answerers,
    get_answerer_spec,
)

__all__ = [
    "AnswerResult",
    "ConversationSample",
    "QAInput",
    "UsageStats",
    "dump_jsonl",
    "iter_conversations",
    "load_jsonl",
    "split_samples",
    "DatasetConfig",
    "DatasetSpec",
    "list_datasets",
    "get_dataset",
    "load_dataset",
    "evaluate_samples",
    "Answerer",
    "PydanticAgentAnswerer",
    "default_prompt_renderer",
    "build_answerer",
    "list_answerers",
    "get_answerer_spec",
]
