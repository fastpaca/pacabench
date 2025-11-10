"""Configuration for datasets and answerers."""

from memharness.answerers.long_context import long_context_answerer
from memharness.datasets.membench import load_membench

# =============================================================================
# Datasets (loader functions returning pydantic-evals Dataset)
# =============================================================================

DATASETS = {
    "membench": lambda **kwargs: load_membench(**kwargs),
    "membench-third": lambda **kwargs: load_membench(agent_type="ThirdAgent", **kwargs),
}

# =============================================================================
# Answerers (task callables for pydantic-evals)
# =============================================================================

ANSWERERS = {
    # Anthropic models
    "claude-opus-long-context": long_context_answerer(
        model="claude-opus-4-20250514",
        provider="anthropic",
    ),
    "claude-sonnet-long-context": long_context_answerer(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
    ),
    # OpenAI models
    "gpt4-long-context": long_context_answerer(
        model="gpt-4o",
        provider="openai",
    ),
    "gpt4-mini-long-context": long_context_answerer(
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
    "lm-studio-minimax-m2": long_context_answerer(
        model="minimax/minimax-m2",
        provider="openai",
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
    ),
}

# =============================================================================
# Adding New Configs
# =============================================================================
#
# ANSWERERS["my-custom"] = long_context_answerer(
#     model="qwen-72b",
#     provider="openai",
#     base_url="http://localhost:8000/v1",
#     api_key="your-key",
# )
#
