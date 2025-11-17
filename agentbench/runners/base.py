"""Base runner abstractions."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentbench.types import Case, EvalContext


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
    metrics: RunnerMetrics = Field(..., description="Performance metrics")

    model_config = {"extra": "forbid"}


class Runner(Protocol):
    """Protocol for runners that execute test cases."""

    async def run_case(self, case: Case, ctx: EvalContext) -> RunnerOutput:
        """
        Execute a test case and return the result.

        Args:
            case: Test case to execute
            ctx: Evaluation context

        Returns:
            RunnerOutput with output, error, and metrics
        """
        ...
