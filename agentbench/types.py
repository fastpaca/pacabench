"""Common types used across the evaluation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agentbench.datasets.base import EvaluationResult
from agentbench.runners.base import RunnerMetrics

if TYPE_CHECKING:
    from agentbench.datasets.base import Dataset
    from agentbench.proxy import ProxyServer
    from agentbench.results import Results
    from agentbench.runners.base import Runner
else:
    Dataset = None
    Runner = None
    ProxyServer = None
    Results = None


class EvalContext(BaseModel):
    """Evaluation context - shared state across pipeline stages."""

    dataset: Dataset = Field(..., description="Dataset instance")
    runner: Runner = Field(..., description="Runner instance")
    results: Results = Field(..., description="Results container for this run")
    judge_model: str = Field("gpt-4o-mini", description="Judge model name")
    judge_client: AsyncOpenAI = Field(..., description="OpenAI client for judge")
    model: str = Field(..., description="Model name")
    openai_api_key: str = Field(..., description="OpenAI API key")
    run_id: str = Field(..., description="Run identifier")
    proxy: ProxyServer = Field(..., description="Proxy server instance")
    proxy_port: int = Field(..., description="Proxy server port")
    embedding_model: str | None = Field(None, description="Embedding model name if applicable")

    model_config = {"arbitrary_types_allowed": True}


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
