"""Evaluation context - shared state across pipeline stages."""

from dataclasses import dataclass

from openai import AsyncOpenAI

from agentbench.proxy import ProxyServer


@dataclass
class EvalContext:
    """Evaluation context carrying shared state across pipeline stages."""

    runner_path: str
    model: str
    openai_api_key: str
    run_id: str
    dataset: str
    proxy: ProxyServer
    proxy_port: int
    judge_client: AsyncOpenAI
    judge_model: str = "gpt-4o-mini"
    embedding_model: str | None = None
