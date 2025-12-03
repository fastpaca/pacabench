"""PacaBench - Benchmark harness for LLM agents.

For the main harness, import from pacabench.engine:
    from pacabench.engine import Harness

For models, import from pacabench.models:
    from pacabench.models import BenchmarkConfig, Case, CaseResult
"""

__version__ = "0.1.0"

# Re-export commonly used models for convenience (these are lightweight)
from pacabench.models import (
    AgentConfig,
    AggregatedMetrics,
    BenchmarkConfig,
    Case,
    CaseResult,
    DatasetConfig,
    ErrorType,
    EvaluationResult,
    EvaluatorConfig,
    GlobalConfig,
    OutputConfig,
    ProxyConfig,
    RunMetadata,
    RunnerMetrics,
    RunnerOutput,
    RunSummary,
    load_config,
)

__all__ = [
    # Models
    "AgentConfig",
    "AggregatedMetrics",
    "BenchmarkConfig",
    "Case",
    "CaseResult",
    "DatasetConfig",
    "ErrorType",
    "EvaluationResult",
    "EvaluatorConfig",
    "GlobalConfig",
    "OutputConfig",
    "ProxyConfig",
    "RunMetadata",
    "RunnerMetrics",
    "RunnerOutput",
    "RunSummary",
    "load_config",
]
