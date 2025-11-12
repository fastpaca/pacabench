"""Configuration for datasets, models, and executors."""

from functools import partial

from memharness.answerers.long_context import long_context_answerer
from memharness.datasets.longmemeval import LongMemEval, load_longmemeval
from memharness.datasets.membench import MemBench, load_membench
from memharness.datasets.webarena import WebArena

# Models (provider + model name configuration)

MODELS = {
    # Anthropic models
    "claude-sonnet-4-5": {"model": "claude-sonnet-4-5", "provider": "anthropic"},
    "claude-haiku-4-5": {"model": "claude-haiku-4-5", "provider": "anthropic"},
    # OpenAI models
    "gpt-4o": {"model": "gpt-4o", "provider": "openai"},
    "gpt-4o-mini": {"model": "gpt-4o-mini", "provider": "openai"},
    # Local models
    "lm-studio-qwen3-30b": {
        "model": "qwen/qwen3-30b-a3b-2507",
        "provider": "openai",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
    },
}

# Datasets (class instances with executor configuration)

DATASETS = {
    # MemBench Q&A datasets
    "membench": MemBench(),
    "membench-third": MemBench(agent_type="ThirdAgent"),
    # LongMemEval Q&A datasets
    "longmemeval-s": LongMemEval(split="s_cleaned"),
    "longmemeval-m": LongMemEval(split="m_cleaned"),
    # WebArena browser-based datasets
    "webarena": WebArena(),
    "webarena-shopping": WebArena(domain="shopping"),
    "webarena-reddit": WebArena(domain="reddit"),
    "webarena-gitlab": WebArena(domain="gitlab"),
}

# Legacy ANSWERERS for backward compatibility (deprecated)
# TODO: Remove after migrating all CLI code to use MODELS + DATASETS

ANSWERERS = {
    # Anthropic models
    "claude-haiku-4-5-long-context": long_context_answerer(
        model="claude-haiku-4-5",
        provider="anthropic",
    ),
    # OpenAI models
    "gpt-4o-long-context": long_context_answerer(
        model="gpt-4o",
        provider="openai",
    ),
    "gpt-4o-mini-long-context": long_context_answerer(
        model="gpt-4o-mini",
        provider="openai",
    ),
    # Local models via LM Studio
    "lm-studio-qwen3-30b": long_context_answerer(
        model="qwen/qwen3-30b-a3b-2507",
        provider="openai",
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
    ),
}

# Legacy DATASETS_LEGACY for backward compatibility
DATASETS_LEGACY = {
    "membench": load_membench,
    "membench-third": partial(load_membench, agent_type="ThirdAgent"),
    "longmemeval-s": partial(load_longmemeval, split="s_cleaned"),
    "longmemeval-m": partial(load_longmemeval, split="m_cleaned"),
}
