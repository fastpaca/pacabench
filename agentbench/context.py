"""Evaluation context for pipeline execution."""

from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agentbench.datasets.base import Dataset
from agentbench.proxy import ProxyServer
from agentbench.results import Results
from agentbench.runners.base import Runner


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
