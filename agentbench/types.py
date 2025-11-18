"""Common types used across the evaluation pipeline."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class RunnerMetrics(BaseModel):
    """Performance metrics collected by runners."""

    model_duration_ms: float = Field(..., description="Model execution duration in milliseconds")
    llm_metrics: dict[str, Any] = Field(
        ..., description="LLM proxy metrics (tokens, latency, cost, etc.)"
    )

    model_config = {"extra": "forbid"}


class RunnerOutput(BaseModel):
    """Output from a runner execution."""

    output: str | None = Field(None, description="Model output")
    error: str | None = Field(None, description="Error message if execution failed")
    duration_ms: float = Field(..., description="Runner execution duration in milliseconds")

    model_config = {"extra": "forbid"}


class EvaluationResult(BaseModel):
    """Results from dataset evaluation."""

    passed: bool = Field(..., description="Whether the case passed")
    f1_score: float | None = Field(None, description="F1 score if applicable")
    f1_passed: bool | None = Field(None, description="Whether F1 evaluation passed")
    judge_passed: bool | None = Field(None, description="Whether judge evaluation passed")

    model_config = {"extra": "forbid"}


class Case(BaseModel):
    """Canonical test case input."""

    id: str = Field(..., description="Unique identifier for the test case")
    task_type: str = Field(..., description="Type of task (e.g., 'qa', 'agentic')")
    inputs: dict[str, Any] = Field(..., description="Input data for the case")
    expected_output: str = Field(..., description="Expected output/answer")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "forbid"}


class JudgeMetrics(BaseModel):
    """Metrics from LLM-as-judge evaluation."""

    input_tokens: int = Field(..., description="Judge input tokens")
    output_tokens: int = Field(..., description="Judge output tokens")

    model_config = {"extra": "forbid"}


class RunnerContext(BaseModel):
    """Lightweight context for runner execution - only runtime config."""

    model: str = Field(..., description="Model name")
    proxy_port: int = Field(..., description="Proxy server port")
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str | None = Field(None, description="Embedding model name if applicable")
    case_id: str = Field(..., description="Case ID for metrics tracking")
    worker_id: int | None = Field(None, description="Worker ID for per-worker resource reuse")

    model_config = {"extra": "forbid"}


class CaseResult(BaseModel):
    """Complete result for a test case (pipeline-level combination)."""

    case_id: str = Field(..., description="Case identifier")
    output: str | None = Field(None, description="Model output")
    error: str | None = Field(None, description="Error message if any")
    metrics: RunnerMetrics = Field(..., description="Performance metrics from runner")
    evaluation: EvaluationResult = Field(..., description="Evaluation results from dataset")
    judge_metrics: JudgeMetrics | None = Field(
        None, description="Judge token metrics if judge was used"
    )

    model_config = {"extra": "forbid"}


class Runner(Protocol):
    """Protocol for runners that execute test cases."""

    async def run_case(self, case: Case, ctx: RunnerContext) -> RunnerOutput:
        """
        Execute a test case and return the result.

        Args:
            case: Test case to execute
            ctx: Runner execution context

        Returns:
            RunnerOutput with output, error, and metrics
        """
        ...
