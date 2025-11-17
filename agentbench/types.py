"""Common types used across the evaluation pipeline."""

from typing import Any

from pydantic import BaseModel, Field

from agentbench.datasets.base import EvaluationResult
from agentbench.runners.base import RunnerMetrics


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
