"""Configuration for datasets and answerers."""

from memharness.answerers.long_context import long_context_answerer
from memharness.datasets.membench import load_membench

# Datasets (loader functions returning pydantic-evals Dataset)

DATASETS = {
    "membench": lambda **kwargs: load_membench(**kwargs),
    "membench-third": lambda **kwargs: load_membench(agent_type="ThirdAgent", **kwargs),
}

# Answerers (task callables for pydantic-evals)

ANSWERERS = {
    # Anthropic models
    "claude-sonnet-long-context": long_context_answerer(
        model="claude-4-5-sonnet",
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