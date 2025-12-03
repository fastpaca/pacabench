"""Case-related models for PacaBench."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    """Represents a single test case input to the agent."""

    case_id: str
    dataset_name: str
    input: str
    expected: str | None = None
    history: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunnerMetrics(BaseModel):
    """Metrics returned by the agent runner (optional override/supplement to proxy)."""

    call_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None


class ErrorType(str, Enum):
    NONE = "none"
    TASK = "task_failure"
    SYSTEM = "system_failure"
    FATAL = "fatal_error"


class RunnerOutput(BaseModel):
    """Raw output from the agent runner process."""

    output: str | None = None
    error: str | None = None
    metrics: RunnerMetrics | None = None
    duration_ms: float = 0.0
    error_type: ErrorType = ErrorType.NONE
    error_traceback: str | None = None
    retry_count: int = 0


class EvaluationResult(BaseModel):
    """Result of an evaluation (judge or heuristic)."""

    passed: bool
    score: float  # 0.0 to 1.0
    reason: str | None = None
    evaluator_latency_ms: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """Final consolidated result for a single case."""

    case_id: str
    dataset_name: str
    agent_name: str
    passed: bool
    output: str | None = None
    error: str | None = None
    error_type: ErrorType = ErrorType.NONE

    # Performance metrics
    runner_duration_ms: float = 0.0
    llm_metrics: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    attempt: int = 1
    timestamp: str | None = None

    # Evaluation details
    f1_score: float | None = None
    f1_passed: bool | None = None
    judge_passed: bool | None = None
    judge_reason: str | None = None
    judge_metrics: dict[str, Any] = Field(default_factory=dict)
    judge_cost_usd: float | None = None

    # For serialization flexibility
    extra: dict[str, Any] = Field(default_factory=dict)
