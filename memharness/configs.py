"""Configuration for datasets, models, and executors."""

from memharness.answerers.gaia_agent import gaia_answerer
from memharness.answerers.long_context import long_context_answerer
from memharness.datasets.gaia import GAIA
from memharness.datasets.longmemeval import LongMemEval
from memharness.datasets.membench import MemBench

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
    # GAIA datasets (multi-step reasoning + tool use)
    "gaia": GAIA(level="all", split="validation"),
    "gaia-level1": GAIA(level="level1", split="validation"),
    "gaia-level2": GAIA(level="level2", split="validation"),
    "gaia-level3": GAIA(level="level3", split="validation"),
}

# Answerers (task callables for pydantic-evals compatibility)
# Maps config name -> model/provider for creating task functions

ANSWERERS = {
    # Long-context answerers (for membench, longmemeval)
    "claude-sonnet-4-5-long-context": long_context_answerer(
        model="claude-sonnet-4-5",
        provider="anthropic",
    ),
    "claude-haiku-4-5-long-context": long_context_answerer(
        model="claude-haiku-4-5",
        provider="anthropic",
    ),
    "gpt-4o-long-context": long_context_answerer(
        model="gpt-4o",
        provider="openai",
    ),
    "gpt-4o-mini-long-context": long_context_answerer(
        model="gpt-4o-mini",
        provider="openai",
    ),
    "lm-studio-qwen3-30b-long-context": long_context_answerer(
        model="qwen/qwen3-30b-a3b-2507",
        provider="openai",
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
    ),
    # GAIA agentic answerers (for complex multi-step tasks with tools)
    "claude-sonnet-4-5-gaia": gaia_answerer(
        model="claude-sonnet-4-5",
        provider="anthropic",
        max_steps=15,
    ),
    "gpt-4o-gaia": gaia_answerer(
        model="gpt-4o",
        provider="openai",
        max_steps=15,
    ),
    "gpt-4o-mini-gaia": gaia_answerer(
        model="gpt-4o-mini",
        provider="openai",
        max_steps=15,
    ),
}

# Config name to model mapping for cost calculation
CONFIG_TO_MODEL = {
    # Long-context configs
    "claude-sonnet-4-5-long-context": "claude-sonnet-4-5",
    "claude-haiku-4-5-long-context": "claude-haiku-4-5",
    "gpt-4o-long-context": "gpt-4o",
    "gpt-4o-mini-long-context": "gpt-4o-mini",
    "lm-studio-qwen3-30b-long-context": "qwen/qwen3-30b-a3b-2507",
    # GAIA agentic configs
    "claude-sonnet-4-5-gaia": "claude-sonnet-4-5",
    "gpt-4o-gaia": "gpt-4o",
    "gpt-4o-mini-gaia": "gpt-4o-mini",
    # Simplified config names
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-haiku-4-5": "claude-haiku-4-5",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}
