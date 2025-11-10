"""Configuration for datasets and answerers.

Direct instantiation of datasets and answerers.
Answerers use lazy initialization internally to avoid requiring API keys on import.
"""

import os

from memharness.answerers.long_context import LongContextAnswerer
from memharness.answerers.mem0 import Mem0Answerer
from memharness.answerers.zep import ZepAnswerer
from memharness.datasets.locomo import LocomoDataset
from memharness.datasets.longmemeval import LongMemEvalDataset
from memharness.datasets.membench import MemBenchDataset

# =============================================================================
# Datasets
# =============================================================================

DATASETS = {
    "membench": MemBenchDataset(),
    "membench-third": MemBenchDataset(agent_type="ThirdAgent"),
    "locomo": LocomoDataset(),
    "longmemeval": LongMemEvalDataset(),
}

# =============================================================================
# Answerers
# =============================================================================

ANSWERERS = {
    # Anthropic models
    "claude-opus-long-context": LongContextAnswerer(
        model="claude-opus-4-20250514",
        provider="anthropic",
    ),
    "claude-sonnet-long-context": LongContextAnswerer(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
    ),
    # OpenAI models
    "gpt4-long-context": LongContextAnswerer(
        model="gpt-4o",
        provider="openai",
    ),
    "gpt4-mini-long-context": LongContextAnswerer(
        model="gpt-4o-mini",
        provider="openai",
    ),
    # Local model example
    "local-long-context": LongContextAnswerer(
        model="local-model",
        provider="openai",
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
    ),
}

# Memory service answerers (if API keys available)
if os.getenv("ZEP_API_KEY"):
    ANSWERERS["zep-claude"] = ZepAnswerer(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        zep_api_key=os.getenv("ZEP_API_KEY"),
    )
    ANSWERERS["zep-gpt4"] = ZepAnswerer(
        model="gpt-4o",
        provider="openai",
        zep_api_key=os.getenv("ZEP_API_KEY"),
    )

if os.getenv("MEM0_API_KEY"):
    ANSWERERS["mem0-claude"] = Mem0Answerer(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        mem0_api_key=os.getenv("MEM0_API_KEY"),
    )
    ANSWERERS["mem0-gpt4"] = Mem0Answerer(
        model="gpt-4o",
        provider="openai",
        mem0_api_key=os.getenv("MEM0_API_KEY"),
    )

# =============================================================================
# Adding New Configs
# =============================================================================
#
# To add a new config, simply instantiate and add to the dicts:
#
# ANSWERERS["my-custom-model"] = LongContextAnswerer(
#     model="qwen-72b",
#     provider="openai",
#     base_url="http://localhost:8000/v1",
#     api_key="your-key",
# )
#
# DATASETS["my-dataset"] = MemBenchDataset(
#     data_dir="/path/to/data",
#     split="all",
# )
#
