"""Metrics models for PacaBench."""

from pydantic import BaseModel


class AggregatedMetrics(BaseModel):
    """Aggregated metrics for a run or subset."""

    accuracy: float = 0.0
    precision: float = 0.0
    total_cases: int = 0
    failed_cases: int = 0

    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0

    avg_llm_latency_ms: float = 0.0
    p50_llm_latency_ms: float = 0.0
    p95_llm_latency_ms: float = 0.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_judge_cost_usd: float = 0.0
