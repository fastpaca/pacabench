"""Pydantic models for PacaBench.

All data models are defined here. Import from this package for all serialization.
"""

from pacabench.models.case import (
    Case,
    CaseResult,
    ErrorType,
    EvaluationResult,
    RunnerMetrics,
    RunnerOutput,
)
from pacabench.models.config import (
    AgentConfig,
    BenchmarkConfig,
    DatasetConfig,
    EvaluatorConfig,
    GlobalConfig,
    OutputConfig,
    ProxyConfig,
    load_config,
)
from pacabench.models.metrics import AggregatedMetrics
from pacabench.models.run import (
    AgentMetadata,
    DatasetMetadata,
    RunMetadata,
    RunStatus,
    RunSummary,
)

__all__ = [
    # case.py
    "Case",
    "CaseResult",
    "ErrorType",
    "EvaluationResult",
    "RunnerMetrics",
    "RunnerOutput",
    # config.py
    "AgentConfig",
    "BenchmarkConfig",
    "DatasetConfig",
    "EvaluatorConfig",
    "GlobalConfig",
    "OutputConfig",
    "ProxyConfig",
    "load_config",
    # metrics.py
    "AggregatedMetrics",
    # run.py
    "AgentMetadata",
    "DatasetMetadata",
    "RunMetadata",
    "RunStatus",
    "RunSummary",
]
